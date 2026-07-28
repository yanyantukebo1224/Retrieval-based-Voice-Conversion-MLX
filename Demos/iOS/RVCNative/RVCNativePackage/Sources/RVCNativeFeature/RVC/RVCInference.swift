import Foundation
import MLX
import MLXRandom
import MLXNN

@MainActor
public class RVCInference: ObservableObject {
    public static let bundle = Bundle.module
    @Published public var status: String = "Idle"
    
    public var onLog: ((String) -> Void)?
    
    var hubertModel: HubertModel?
    var synthesizer: Synthesizer?
    var rmvpe: RMVPE?
    var crepe: CREPE?
    var dspPitch: DSPPitchExtractor?
    var indexManager: IndexManager?
    var indexRate: Float = 0.75
    var modelSampleRate: Int = 40000
    
    private func log(_ message: String) {
        print(message)
        DispatchQueue.main.async {
            self.onLog?(message)
        }
    }
    
    public init() {
        log("RVCInference: Initializing...")
        #if targetEnvironment(simulator)
        MLX.Device.setDefault(device: Device.cpu)
        log("RVCInference: Running on Simulator, forced CPU device.")
        #endif
    }

    public func unloadModels() {
        log("RVCInference: Unloading models...")
        hubertModel = nil
        synthesizer = nil
        rmvpe = nil
        crepe = nil
        dspPitch = nil
        indexManager?.unload()
        indexManager = nil
        status = "Idle"
        log("RVCInference: Models unloaded.")
    }

    public func loadIndex(url: URL, rate: Float = 0.75) throws {
        log("RVCInference: Loading index from \(url.lastPathComponent)")
        let manager = IndexManager()
        
        try manager.load(url: url, logger: { [weak self] msg in
            self?.log(msg)
        })
        
        self.indexManager = manager
        self.indexRate = rate
        log("RVCInference: Index loaded with \(manager.count) vectors, rate=\(rate)")
    }

    public func unloadIndex() {
        indexManager?.unload()
        indexManager = nil
        log("RVCInference: Index unloaded.")
    }

    public func loadWeights(hubertURL: URL, modelURL: URL, rmvpeURL: URL? = nil, crepeURL: URL? = nil) async throws {
        let modelName = modelURL.lastPathComponent.lowercased()
        if modelName.contains("rmvpe") || modelName.contains("hubert") {
            log("RVCInference: ⚠️ [GUARD] Invalid modelURL (\(modelURL.lastPathComponent)). RMVPE/HuBERT cannot be loaded as Synthesizer!")
            return
        }

        DispatchQueue.main.async { self.status = "Loading models..." }
        
        // 1. Load Hubert
        log("RVCInference: Loading Hubert from \(hubertURL.lastPathComponent)")
        var actualHubertURL = hubertURL
        let hExt = hubertURL.pathExtension.lowercased()
        if hExt == "pt" || hExt == "pth" {
            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let dest = docs.appendingPathComponent("hubert_base.safetensors")
            if FileManager.default.fileExists(atPath: dest.path) {
                actualHubertURL = dest
            } else if let arrays = try? PthConverter.shared.convert(url: hubertURL), !arrays.isEmpty {
                try? MLX.save(arrays: arrays, url: dest)
                actualHubertURL = dest
            }
        }
        
        let hubertWeights = try MLX.loadArrays(url: actualHubertURL)
        log("RVCInference: Raw HuBERT file keys count: \(hubertWeights.count)")

        self.hubertModel = HubertModel(config: HubertConfig())
        var newParams: [String: MLXArray] = [:]
        for (k, v) in hubertWeights {
            var newKey = k
            var val = v
            
            while newKey.hasPrefix("model.") || newKey.hasPrefix("hubert.") {
                if newKey.hasPrefix("model.") { newKey = String(newKey.dropFirst(6)) }
                else if newKey.hasPrefix("hubert.") { newKey = String(newKey.dropFirst(7)) }
            }

            if newKey.hasPrefix("encoder.layers.") {
                let parts = newKey.components(separatedBy: ".")
                if parts.count >= 3, let idx = Int(parts[2]) {
                    newKey = "encoder.l\(idx)." + parts.dropFirst(3).joined(separator: ".")
                }
            }
            
            if newKey.hasPrefix("feature_extractor.conv_layers.") {
                let parts = newKey.components(separatedBy: ".")
                if parts.count >= 3, let idx = Int(parts[2]) {
                    let subPath = parts.dropFirst(3).joined(separator: ".")
                    if subPath == "0.weight" { newKey = "feature_extractor.l\(idx).conv.weight" }
                    else if subPath == "0.bias" { newKey = "feature_extractor.l\(idx).conv.bias" }
                    else if subPath == "2.weight" || subPath == "1.weight" { newKey = "feature_extractor.l\(idx).layer_norm.weight" }
                    else if subPath == "2.bias" || subPath == "1.bias" { newKey = "feature_extractor.l\(idx).layer_norm.bias" }
                    else { newKey = "feature_extractor.l\(idx)." + subPath }
                }
            }
            
            if newKey.hasSuffix(".weight") && val.ndim == 3 {
                val = val.transposed(axes: [0, 2, 1])
            }
            newParams[newKey] = val
        }
        
        let gKey = "encoder.pos_conv_embed.conv.weight_g"
        let vKey = "encoder.pos_conv_embed.conv.weight_v"
        let outKey = "encoder.pos_conv_embed.conv.weight"
        
        if let weight_g_raw = newParams[gKey], let weight_v_raw = newParams[vKey] {
             let weight_v: MLXArray
             let weight_g: MLXArray
             if weight_v_raw.shape[1] < weight_v_raw.shape[2] {
                 weight_v = weight_v_raw.transposed(axes: [0, 2, 1])
                 weight_g = weight_g_raw.transposed(axes: [0, 2, 1])
             } else {
                 weight_v = weight_v_raw
                 weight_g = weight_g_raw
             }
             
             let v_sqr = weight_v * weight_v
             let v_sum = v_sqr.sum(axes: [0, 2], keepDims: true)
             let v_norm = sqrt(v_sum + 1e-12)
             let weight_normalized = weight_v / v_norm
             let weight_fused = weight_g * weight_normalized
             
             newParams[outKey] = weight_fused
             newParams.removeValue(forKey: gKey)
             newParams.removeValue(forKey: vKey)
        }
        
        self.hubertModel?.update(parameters: ModuleParameters.unflattened(newParams))
        
        // 2. Load Synthesizer
        log("RVCInference: Loading Synthesizer from \(modelURL.lastPathComponent)")
        let modelWeights = try MLX.loadArrays(url: modelURL)

        var detectedSR = self.modelSampleRate
        var detectedUpsRates = [10, 10, 2, 2]
        var detectedKernelSizes = [16, 16, 4, 4]

        let configURL = modelURL.deletingPathExtension().appendingPathExtension("json")
        if FileManager.default.fileExists(atPath: configURL.path) {
            if let data = try? Data(contentsOf: configURL),
               let json = try? JSONSerialization.jsonObject(with: data) as? [Any] {
                if json.count > 17, let sr = json[17] as? Int { detectedSR = sr }
                if json.count > 12, let uArr = json[12] as? [Any] {
                    let u = uArr.compactMap { $0 as? Int }
                    if !u.isEmpty { detectedUpsRates = u }
                }
                if json.count > 14, let kArr = json[14] as? [Any] {
                    let k = kArr.compactMap { $0 as? Int }
                    if !k.isEmpty { detectedKernelSizes = k }
                }
            }
        }

        self.synthesizer = Synthesizer(
            interChannels: 192,
            hiddenChannels: 192,
            filterChannels: 768,
            nHeads: 2,
            nLayers: 6,
            kernelSize: 3,
            pDropout: 0.0,
            embeddingDim: 768,
            speakerEmbedDim: 256,
            ginChannels: 256,
            useF0: true,
            upsampleRates: detectedUpsRates,
            upsampleKernelSizes: detectedKernelSizes,
            sampleRate: detectedSR
        )
        self.modelSampleRate = detectedSR
        
        func needsConvInsertion(_ key: String) -> Bool {
            if key.hasPrefix("dec.conv_pre.") || key.hasPrefix("dec.conv_post.") { return true }
            if key.contains("dec.noise_conv_") { return true }
            if key.contains("dec.resblock_") && (key.contains(".c1_") || key.contains(".c2_")) { return true }
            return false
        }

        var synthParams: [String: MLXArray] = [:]
        for (k, v) in modelWeights {
            var newK = k
            var newV = v

            if newK.contains("dec.ups.") { newK = newK.replacingOccurrences(of: "dec.ups.", with: "dec.up_") }
            if newK.contains("dec.noise_convs.") { newK = newK.replacingOccurrences(of: "dec.noise_convs.", with: "dec.noise_conv_") }
            if newK.contains("dec.resblocks.") {
                newK = newK.replacingOccurrences(of: "dec.resblocks.", with: "dec.resblock_")
                newK = newK.replacingOccurrences(of: ".convs1.", with: ".c1_")
                newK = newK.replacingOccurrences(of: ".convs2.", with: ".c2_")
            }

            if newK.contains("dec.resblock_") && (newK.contains(".c1_") || newK.contains(".c2_")) {
                if newK.hasSuffix(".weight") && !newK.contains(".conv.") {
                    newK = newK.replacingOccurrences(of: ".weight", with: ".conv.weight")
                }
                if newK.hasSuffix(".bias") && !newK.contains(".conv.") {
                    newK = newK.replacingOccurrences(of: ".bias", with: ".conv.bias")
                }
            }


            if needsConvInsertion(newK) && !newK.contains(".conv.") {
                if newK.hasSuffix(".weight") { newK = String(newK.dropLast(7)) + ".conv.weight" }
                else if newK.hasSuffix(".bias") { newK = String(newK.dropLast(5)) + ".conv.bias" }
            }

            if newK.hasSuffix(".weight") && newV.ndim == 3 {
                if newK.contains(".up_") || newK.contains(".ups.") {
                    newV = newV.transposed(axes: [1, 2, 0])
                } else {
                    newV = newV.transposed(axes: [0, 2, 1])
                }
            }

            synthParams[newK] = newV
        }

        for i in 0..<4 {
            let gKey = "dec.up_\(i).weight_g"
            let vKey = "dec.up_\(i).weight_v"
            let outKey = "dec.up_\(i).weight"
            
            if let weight_g = synthParams[gKey], let weight_v = synthParams[vKey] {
                let v_sqr = weight_v * weight_v
                let v_sum = v_sqr.sum(axes: [0, 2], keepDims: true)
                let v_norm = sqrt(v_sum + 1e-12)
                let weight_normalized = weight_v / v_norm
                let weight_fused = weight_g * weight_normalized
                
                synthParams[outKey] = weight_fused.transposed(axes: [1, 2, 0])
                synthParams.removeValue(forKey: gKey)
                synthParams.removeValue(forKey: vKey)
            }
        }
        
        for i in 0..<12 {
            for (convPrefix, convCount) in [("c1_", 3), ("c2_", 3)] {
                for j in 0..<convCount {
                    let base = "dec.resblock_\(i).\(convPrefix)\(j)"
                    let gKey = "\(base).weight_g"
                    let vKey = "\(base).weight_v"
                    let outKey = "\(base).conv.weight"
                    
                    if let weight_g = synthParams[gKey], let weight_v = synthParams[vKey] {
                        let v_sqr = weight_v * weight_v
                        let v_sum = v_sqr.sum(axes: [1, 2], keepDims: true)
                        let v_norm = sqrt(v_sum + 1e-12)
                        let weight_normalized = weight_v / v_norm
                        let weight_fused = weight_g * weight_normalized
                        
                        synthParams[outKey] = weight_fused.transposed(axes: [0, 2, 1])
                        synthParams.removeValue(forKey: gKey)
                        synthParams.removeValue(forKey: vKey)
                    }
                }
            }
        }

        self.synthesizer?.update(parameters: ModuleParameters.unflattened(synthParams))
        self.synthesizer?.train(false)
        
        // 3. Load RMVPE
        if let rmvpeURL = rmvpeURL {
            do {
                let rmvpeWeights = try MLX.loadArrays(url: rmvpeURL)
                self.rmvpe = RMVPE()
                
                var remappedRMVPE: [String: MLXArray] = [:]
                for (k, v) in rmvpeWeights {
                    var newKey = k
                    if newKey.contains("num_batches_tracked") { continue }
                    if newKey.hasPrefix("fc.bigru.forward_grus.0.") {
                        newKey = "bigru.fwd0." + String(newKey.dropFirst("fc.bigru.forward_grus.0.".count))
                    }
                    if newKey.hasPrefix("fc.bigru.backward_grus.0.") {
                        newKey = "bigru.bwd0." + String(newKey.dropFirst("fc.bigru.backward_grus.0.".count))
                    }
                    if newKey.hasPrefix("fc.linear.") {
                        newKey = "linear." + String(newKey.dropFirst("fc.linear.".count))
                    }
                    
                    newKey = newKey.replacingOccurrences(of: ".blocks.0.", with: ".b0.")
                    newKey = newKey.replacingOccurrences(of: ".blocks.1.", with: ".b1.")
                    newKey = newKey.replacingOccurrences(of: ".blocks.2.", with: ".b2.")
                    newKey = newKey.replacingOccurrences(of: ".blocks.3.", with: ".b3.")
                    
                    if newKey.contains(".layers.") {
                        newKey = newKey.replacingOccurrences(of: ".layers.0.", with: ".l0.")
                        newKey = newKey.replacingOccurrences(of: ".layers.1.", with: ".l1.")
                        newKey = newKey.replacingOccurrences(of: ".layers.2.", with: ".l2.")
                        newKey = newKey.replacingOccurrences(of: ".layers.3.", with: ".l3.")
                        newKey = newKey.replacingOccurrences(of: ".layers.4.", with: ".l4.")
                    }
                    
                    newKey = newKey.replacingOccurrences(of: ".running_mean", with: ".runningMean")
                    newKey = newKey.replacingOccurrences(of: ".running_var", with: ".runningVar")

                    var val = v
                    if newKey.hasSuffix(".weight") {
                        if val.ndim == 3 {
                            val = val.transposed(axes: [0, 2, 1])
                        } else if val.ndim == 4 {
                            val = val.transposed(axes: [0, 2, 3, 1])
                        }
                    }
                    remappedRMVPE[newKey] = val
                }

                self.rmvpe?.update(parameters: ModuleParameters.unflattened(remappedRMVPE))
                self.rmvpe?.setTrainingMode(false)
                log("RVCInference: ✅ RMVPE loaded")
            } catch {
                log("RVCInference: Failed to load RMVPE: \(error)")
                self.rmvpe = nil
            }
        }

        if let crepeURL = crepeURL {
            self.crepe = try? CREPE(weightsURL: crepeURL, modelType: "full")
        }

        self.hubertModel?.train(false)
        DispatchQueue.main.async { self.status = "Models Loaded" }
    }
    
    public func infer(
        audioURL: URL,
        outputURL: URL,
        pitchShift: Int = 0,
        f0Method: String = "rmvpe",
        indexRate: Float = 0.75,
        volumeEnvelope: Float = 1.0
    ) async {
        do {
            DispatchQueue.main.async { self.status = "Loading Audio..." }
            let (audioArray, nativeSR) = try AudioProcessor.shared.loadAudio(url: audioURL)
            let totalSamples = audioArray.size
            
            let maxSamples = Int(nativeSR * 30.0)
            var audioToProcess = audioArray
            if totalSamples > maxSamples {
                audioToProcess = audioArray[0..<maxSamples]
            }
            
            audioToProcess = applyButterworthHighPass(audioToProcess)
            let padSamples = Int(nativeSR * 0.1)
            let audioPadded = padReflect(audioToProcess, padding: padSamples)
            
            let outputPadded = try await inferChunk(
                chunk: audioPadded,
                pitchShift: pitchShift,
                f0Method: f0Method,
                indexRate: indexRate
            )
            MLX.eval(outputPadded)
            
            let outputRatio: Float = Float(self.modelSampleRate) / Float(nativeSR)
            let cropSamples = Int(Float(padSamples) * outputRatio)
            let outputLen = outputPadded.shape[1]
            let coreStart = cropSamples
            let coreEnd = outputLen - cropSamples
            
            var finalOutput: MLXArray
            if coreEnd > coreStart && coreEnd <= outputLen {
                finalOutput = outputPadded[0..., coreStart..<coreEnd, 0...].squeezed()
            } else {
                finalOutput = outputPadded.squeezed()
            }
            
            MLX.eval(finalOutput)
            
            // Peak normalization for clear audible volume
            let maxAmp = MLX.abs(finalOutput).max().item(Float.self)
            if maxAmp > 0.001 && maxAmp < 0.95 {
                let targetAmp: Float = 0.85
                finalOutput = finalOutput * (targetAmp / maxAmp)
                MLX.eval(finalOutput)
            }

            if volumeEnvelope != 1.0 {
                finalOutput = finalOutput * volumeEnvelope
                MLX.eval(finalOutput)
            }

            try AudioProcessor.shared.saveAudio(array: finalOutput, url: outputURL, sampleRate: Double(self.modelSampleRate))
            DispatchQueue.main.async { self.status = "Done!" }
            
        } catch {
            log("RVCInference Error: \(error)")
            DispatchQueue.main.async { self.status = "Error: \(error.localizedDescription)" }
        }
    }
    
    private func inferChunk(
        chunk: MLXArray,
        pitchShift: Int,
        f0Method: String,
        indexRate: Float
    ) async throws -> MLXArray {
        var cleanAudio = chunk
        if cleanAudio.ndim == 2 { cleanAudio = cleanAudio.mean(axis: 0) }
        if cleanAudio.ndim != 1 { cleanAudio = cleanAudio.flattened() }
        
        let audioInput = cleanAudio.expandedDimensions(axis: 0)
        
        // 1. HuBERT による言語特徴量抽出
        guard let hubertModel = hubertModel else {
            throw NSError(domain: "RVCInference", code: 1, userInfo: [NSLocalizedDescriptionKey: "HuBERT model not loaded."])
        }
        
        let hubertFeatures: MLXArray = autoreleasepool {
            let feat = hubertModel(audioInput)
            MLX.eval(feat)
            return feat
        }
        GPU.clearCache()
        
        // 2. RMVPE によるピッチ抽出
        var f0: MLXArray
        if let rmvpe = rmvpe {
            f0 = rmvpe.infer(audio: cleanAudio, thred: 0.55)
        } else {
            let frames = hubertFeatures.shape[1] * 2
            f0 = MLX.zeros([1, frames, 1])
        }
        MLX.eval(f0)
        GPU.clearCache()
        
        // 3. 特徴量のフレーム調整 (2倍展開)
        let N = hubertFeatures.shape[0]
        let L = hubertFeatures.shape[1]
        let C = hubertFeatures.shape[2]
        
        let expanded = hubertFeatures.expandedDimensions(axis: 2)
        let repeated = MLX.repeated(expanded, count: 2, axis: 2)
        var phone = repeated.reshaped([N, L * 2, C])
        
        // F0 と phone のフレーム数を最小長に切り揃える
        let phoneLen = phone.shape[1]
        let f0Len = f0.shape[1]
        let minLen = min(phoneLen, f0Len)
        
        if phoneLen != minLen { phone = phone[0..., 0..<minLen, 0...] }
        if f0Len != minLen { f0 = f0[0..., 0..<minLen, 0...] }

        if indexRate > 0, let indexManager = indexManager {
            phone = indexManager.search(features: phone, indexRate: indexRate)
        }

        guard let synth = synthesizer else {
            throw NSError(domain: "RVCInference", code: 2, userInfo: [NSLocalizedDescriptionKey: "Voice model not loaded"])
        }

        var f0Hz = f0.squeezed(axes: [2])
        if pitchShift != 0 {
            let multiplier = pow(2.0, Float(pitchShift) / 12.0)
            f0Hz = f0Hz * multiplier
        }
        
        let f0_min: Float = 50.0
        let f0_max: Float = 1100.0
        let f0HzClean = MLX.where(f0Hz .<= f0_min, MLXArray(0.0), f0Hz)
        
        let f0_mel_min = 1127.0 * Darwin.log(1.0 + Double(f0_min) / 700.0)
        let f0_mel_max = 1127.0 * Darwin.log(1.0 + Double(f0_max) / 700.0)
        let f0_mel = 1127.0 * MLX.log(1.0 + MLX.maximum(f0HzClean, f0_min) / 700.0)
        
        var pitch = (f0_mel - f0_mel_min) * (254.0 / (f0_mel_max - f0_mel_min)) + 1.0
        pitch = MLX.where(f0HzClean .== 0.0, MLXArray(1.0), pitch) 
        pitch = MLX.clip(pitch, min: 1.0, max: 255.0)
        let pitchBuckets = pitch.asType(Int32.self)
        
        let nsff0 = f0HzClean.expandedDimensions(axis: 2)
        let phoneLengths = MLXArray([Int32(minLen)])
        let sid = MLXArray([Int32(0)])

        // 4. Synthesizer 推論実行
        let audioConverted: MLXArray = autoreleasepool {
            let out = synth.infer(
                phone: phone,
                phoneLengths: phoneLengths,
                pitch: pitchBuckets,
                nsff0: nsff0,
                sid: sid
            )
            MLX.eval(out)
            return out
        }
        GPU.clearCache()

        return audioConverted
    }

    public func runBenchmark(audioURL: URL, referenceURL: URL?, outputURL: URL) async throws -> String {
        await infer(audioURL: audioURL, outputURL: outputURL)
        return "Inference completed."
    }
    
    private func applyButterworthHighPass(_ audio: MLXArray) -> MLXArray {
        let b: [Double] = [0.9699606451838447, -4.849803225919223, 9.699606451838447, -9.699606451838447, 4.849803225919223, -0.9699606451838447]
        let a: [Double] = [1.0, -4.939001819168364, 9.757863526739543, -9.639544849413458, 4.761506797356209, -0.9408236532054606]
        
        MLX.eval(audio)
        let samplesFloat = audio.asArray(Float.self)
        let samples = samplesFloat.map { Double($0) }
        
        func filter(_ x: [Double]) -> [Double] {
            var y = [Double](repeating: 0, count: x.count)
            for n in 0..<x.count {
                var val = b[0] * x[n]
                if n > 0 { val += b[1] * x[n-1] }
                if n > 1 { val += b[2] * x[n-2] }
                if n > 2 { val += b[3] * x[n-3] }
                if n > 3 { val += b[4] * x[n-4] }
                if n > 4 { val += b[5] * x[n-5] }
                
                if n > 0 { val -= a[1] * y[n-1] }
                if n > 1 { val -= a[2] * y[n-2] }
                if n > 2 { val -= a[3] * y[n-3] }
                if n > 3 { val -= a[4] * y[n-4] }
                if n > 4 { val -= a[5] * y[n-5] }
                
                y[n] = val
            }
            return y
        }
        
        let y_fwd = filter(samples)
        let y_rev = Array(y_fwd.reversed())
        let y_back = filter(y_rev)
        let y_final = Array(y_back.reversed())
        
        let resultFloat = y_final.map { Float($0) }
        return MLXArray(resultFloat)
    }
    
    private func padReflect(_ audio: MLXArray, padding: Int) -> MLXArray {
        MLX.eval(audio)
        let samples = audio.asArray(Float.self)
        let n = samples.count
        
        guard n > 1 else { return audio }
        
        var leftPad: [Float] = []
        if padding > 0 {
            let start = 1
            let end = min(padding, n - 1)
            if end >= start {
                leftPad = Array(samples[start...end].reversed())
            }
        }
        
        var rightPad: [Float] = []
        if padding > 0 {
            let start = max(0, n - 1 - padding)
            let end = n - 2
            if end >= start {
                rightPad = Array(samples[start...end].reversed())
            }
        }
        
        let result = leftPad + samples + rightPad
        return MLXArray(result)
    }
}
