import Foundation
import MLX
import MLXNN
import MLXRandom

/// RVC Synthesizer Implementation for iOS (MLX Native Shape: B, T, C)

// MARK: - Utility Functions

func sequenceMask(lengths: MLXArray, maxLength: Int) -> MLXArray {
    let x = MLXArray(0..<maxLength).asType(lengths.dtype)
    return (x.expandedDimensions(axis: 0) .< lengths.expandedDimensions(axis: 1)).asType(Float32.self)
}

// MARK: - Multi-Head Attention

class MultiHeadAttention: Module {
    let channels: Int
    let outChannels: Int
    let nHeads: Int
    let kChannels: Int
    let windowSize: Int?
    
    let conv_q: MLXNN.Conv1d
    let conv_k: MLXNN.Conv1d
    let conv_v: MLXNN.Conv1d
    let conv_o: MLXNN.Conv1d
    let drop: MLXNN.Dropout
    
    init(channels: Int, outChannels: Int, nHeads: Int, pDropout: Float = 0.0, windowSize: Int? = nil) {
        self.channels = channels
        self.outChannels = outChannels
        self.nHeads = nHeads
        self.kChannels = channels / nHeads
        self.windowSize = windowSize
        
        self.conv_q = MLXNN.Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: 1)
        self.conv_k = MLXNN.Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: 1)
        self.conv_v = MLXNN.Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: 1)
        self.conv_o = MLXNN.Conv1d(inputChannels: channels, outputChannels: outChannels, kernelSize: 1)
        self.drop = MLXNN.Dropout(p: pDropout)
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray, c: MLXArray, attnMask: MLXArray? = nil) -> MLXArray {
        let q = conv_q(x)
        let k = conv_k(c)
        let v = conv_v(c)
        let (out, _) = attention(query: q, key: k, value: v, mask: attnMask)
        return conv_o(out)
    }
    
    func attention(query: MLXArray, key: MLXArray, value: MLXArray, mask: MLXArray?) -> (MLXArray, MLXArray) {
        let b = key.shape[0]
        let t_s = key.shape[1]
        let t_t = query.shape[1]
        
        var q = query.reshaped([b, t_t, nHeads, kChannels]).transposed(0, 2, 1, 3)
        var k = key.reshaped([b, t_s, nHeads, kChannels]).transposed(0, 2, 1, 3)
        var v = value.reshaped([b, t_s, nHeads, kChannels]).transposed(0, 2, 1, 3)
        
        var scores = MLX.matmul(q / sqrt(Float(kChannels)), k.transposed(0, 1, 3, 2))
        
        if let mask = mask {
            scores = MLX.where(mask .== 0, MLXArray(-1e4), scores)
        }
        
        let pAttn = drop(softmax(scores, axis: -1))
        var output = MLX.matmul(pAttn, v)
        
        output = output.transposed(0, 2, 1, 3).reshaped([b, t_t, -1])
        return (output, pAttn)
    }
}

// MARK: - Feed Forward Network

class FFN: Module {
    let conv_1: MLXNN.Conv1d
    let conv_2: MLXNN.Conv1d
    let drop: MLXNN.Dropout
    let kernel_size: Int
    
    init(inChannels: Int, outChannels: Int, filterChannels: Int, kernelSize: Int, pDropout: Float = 0.0) {
        self.conv_1 = MLXNN.Conv1d(inputChannels: inChannels, outputChannels: filterChannels, kernelSize: kernelSize, padding: 0)
        self.conv_2 = MLXNN.Conv1d(inputChannels: filterChannels, outputChannels: outChannels, kernelSize: kernelSize, padding: 0)
        self.drop = MLXNN.Dropout(p: pDropout)
        self.kernel_size = kernelSize
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray, xMask: MLXArray) -> MLXArray {
        let padTotal = kernel_size - 1
        let padL = padTotal / 2
        let padR = padTotal - padL
        
        var h = x * xMask
        h = MLX.padded(h, widths: [IntOrPair((0, 0)), IntOrPair((padL, padR)), IntOrPair((0, 0))])
        h = conv_1(h)
        h = MLX.maximum(h, MLXArray(0))  // ReLU
        h = drop(h)
        
        h = h * xMask
        h = MLX.padded(h, widths: [IntOrPair((0, 0)), IntOrPair((padL, padR)), IntOrPair((0, 0))])
        h = conv_2(h)
        return h * xMask
    }
}

// MARK: - Encoder (Transformer-style)

class RVCEncoder: Module {
    let nLayers: Int
    let drop: MLXNN.Dropout
    var attn_layers: [MultiHeadAttention] = []
    var norm_layers_1: [MLXNN.LayerNorm] = []
    var ffn_layers: [FFN] = []
    var norm_layers_2: [MLXNN.LayerNorm] = []
    
    init(hiddenChannels: Int, filterChannels: Int, nHeads: Int, nLayers: Int, kernelSize: Int = 1, pDropout: Float = 0.0, windowSize: Int = 10) {
        self.nLayers = nLayers
        self.drop = MLXNN.Dropout(p: pDropout)
        
        self.attn_layers = (0..<nLayers).map { _ in
            MultiHeadAttention(channels: hiddenChannels, outChannels: hiddenChannels, nHeads: nHeads, pDropout: pDropout, windowSize: windowSize)
        }
        self.norm_layers_1 = (0..<nLayers).map { _ in
            MLXNN.LayerNorm(dimensions: hiddenChannels)
        }
        self.ffn_layers = (0..<nLayers).map { _ in
            FFN(inChannels: hiddenChannels, outChannels: hiddenChannels, filterChannels: filterChannels, kernelSize: kernelSize, pDropout: pDropout)
        }
        self.norm_layers_2 = (0..<nLayers).map { _ in
            MLXNN.LayerNorm(dimensions: hiddenChannels)
        }
        
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray, xMask: MLXArray) -> MLXArray {
        let xMaskB = xMask.asType(Float32.self)
        let attnMask = (xMaskB * xMaskB.transposed(0, 2, 1)).expandedDimensions(axis: 1)
        
        var h = x * xMask
        
        for i in 0..<nLayers {
            let y = attn_layers[i](h, c: h, attnMask: attnMask)
            h = norm_layers_1[i](h + drop(y))
            
            let y2 = ffn_layers[i](h, xMask: xMask)
            h = norm_layers_2[i](h + drop(y2))
        }
        
        return h * xMask
    }
}

// MARK: - TextEncoder (enc_p)

class TextEncoder: Module {
    let hiddenChannels: Int
    let outChannels: Int
    let emb_phone: MLXNN.Linear
    let emb_pitch: MLXNN.Embedding?
    let encoder: RVCEncoder
    let proj: MLXNN.Conv1d
    
    init(outChannels: Int, hiddenChannels: Int, filterChannels: Int, nHeads: Int, nLayers: Int, kernelSize: Int, pDropout: Float, embeddingDim: Int, f0: Bool = true) {
        self.hiddenChannels = hiddenChannels
        self.outChannels = outChannels
        
        self.emb_phone = MLXNN.Linear(embeddingDim, hiddenChannels)
        self.emb_pitch = f0 ? MLXNN.Embedding(embeddingCount: 256, dimensions: hiddenChannels) : nil
        self.encoder = RVCEncoder(hiddenChannels: hiddenChannels, filterChannels: filterChannels, nHeads: nHeads, nLayers: nLayers, kernelSize: kernelSize, pDropout: pDropout)
        self.proj = MLXNN.Conv1d(inputChannels: hiddenChannels, outputChannels: outChannels * 2, kernelSize: 1)
        
        super.init()
    }
    
    func callAsFunction(_ phone: MLXArray, pitch: MLXArray?, lengths: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        // phone: (B, T, EmbDim)
        var x = emb_phone(phone)
        
        if let pitch = pitch, let embPitch = emb_pitch {
            let pEmb = embPitch(pitch)
            x = x + pEmb
        }
        
        x = x * sqrt(Float(hiddenChannels))
        x = leakyRelu(x, negativeSlope: 0.1)
        
        let xMask = sequenceMask(lengths: lengths, maxLength: x.shape[1])
        let xMaskExpanded = xMask.expandedDimensions(axis: -1)  // (B, T, 1)
        
        x = encoder(x, xMask: xMaskExpanded)

        // MLX Standard: proj outputs (B, T, outChannels * 2)
        let stats = proj(x) * xMaskExpanded

        // 転置せず Channel 軸（最後のアシス）で 2 分割 (B, T, C)
        let splitIdx = outChannels
        let m = stats[0..., 0..., 0..<splitIdx]         // (B, T, C)
        let logs = stats[0..., 0..., splitIdx...]       // (B, T, C)

        return (m, logs, xMaskExpanded)
    }
}

// MARK: - WaveNet for Flow

class WaveNet: Module {
    let hiddenChannels: Int
    let nLayers: Int
    let cond_layer: MLXNN.Conv1d?

    let in_layer_0: MLXNN.Conv1d
    let in_layer_1: MLXNN.Conv1d
    let in_layer_2: MLXNN.Conv1d

    let res_skip_layer_0: MLXNN.Conv1d
    let res_skip_layer_1: MLXNN.Conv1d
    let res_skip_layer_2: MLXNN.Conv1d

    init(hiddenChannels: Int, kernelSize: Int, dilationRate: Int, nLayers: Int, ginChannels: Int) {
        assert(nLayers == 3, "WaveNet hardcoded for 3 layers")
        self.hiddenChannels = hiddenChannels
        self.nLayers = nLayers

        self.cond_layer = ginChannels != 0 ? MLXNN.Conv1d(inputChannels: ginChannels, outputChannels: 2 * hiddenChannels * nLayers, kernelSize: 1) : nil

        let dilations = (0..<nLayers).map { Int(pow(Double(dilationRate), Double($0))) }
        let paddings = dilations.map { (kernelSize * $0 - $0) / 2 }

        self.in_layer_0 = MLXNN.Conv1d(inputChannels: hiddenChannels, outputChannels: 2 * hiddenChannels, kernelSize: kernelSize, padding: paddings[0], dilation: dilations[0])
        self.in_layer_1 = MLXNN.Conv1d(inputChannels: hiddenChannels, outputChannels: 2 * hiddenChannels, kernelSize: kernelSize, padding: paddings[1], dilation: dilations[1])
        self.in_layer_2 = MLXNN.Conv1d(inputChannels: hiddenChannels, outputChannels: 2 * hiddenChannels, kernelSize: kernelSize, padding: paddings[2], dilation: dilations[2])

        self.res_skip_layer_0 = MLXNN.Conv1d(inputChannels: hiddenChannels, outputChannels: 2 * hiddenChannels, kernelSize: 1)
        self.res_skip_layer_1 = MLXNN.Conv1d(inputChannels: hiddenChannels, outputChannels: 2 * hiddenChannels, kernelSize: 1)
        self.res_skip_layer_2 = MLXNN.Conv1d(inputChannels: hiddenChannels, outputChannels: hiddenChannels, kernelSize: 1)

        super.init()
    }

    func callAsFunction(_ x: MLXArray, xMask: MLXArray, g: MLXArray?) -> MLXArray {
        var h = x
        var outputAcc = MLX.zeros(like: x)

        var gCond: MLXArray? = nil
        if let g = g, let condLayer = cond_layer {
            gCond = condLayer(g)
        }

        let inLayers = [in_layer_0, in_layer_1, in_layer_2]
        let resSkipLayers = [res_skip_layer_0, res_skip_layer_1, res_skip_layer_2]

        for i in 0..<nLayers {
            let xIn = inLayers[i](h)

            var acts: MLXArray
            if let gCond = gCond {
                let startCh = i * 2 * hiddenChannels
                let endCh = (i + 1) * 2 * hiddenChannels
                let gSlice = gCond[0..., 0..., startCh..<endCh]

                let combined = xIn + gSlice
                let tAct = tanh(combined[0..., 0..., 0..<hiddenChannels])
                let sAct = sigmoid(combined[0..., 0..., hiddenChannels...])
                acts = tAct * sAct
            } else {
                let tAct = tanh(xIn[0..., 0..., 0..<hiddenChannels])
                let sAct = sigmoid(xIn[0..., 0..., hiddenChannels...])
                acts = tAct * sAct
            }

            let resSkipActs = resSkipLayers[i](acts)

            if i < nLayers - 1 {
                let resActs = resSkipActs[0..., 0..., 0..<hiddenChannels]
                h = (h + resActs) * xMask
                outputAcc = outputAcc + resSkipActs[0..., 0..., hiddenChannels...]
            } else {
                outputAcc = outputAcc + resSkipActs
            }
        }

        return outputAcc * xMask
    }
}

// MARK: - Residual Coupling Layer

class ResidualCouplingLayer: Module {
    let halfChannels: Int
    let meanOnly: Bool
    let pre: MLXNN.Conv1d
    let enc: WaveNet
    let post: MLXNN.Conv1d
    
    init(channels: Int, hiddenChannels: Int, kernelSize: Int, dilationRate: Int, nLayers: Int, ginChannels: Int, meanOnly: Bool = false) {
        self.halfChannels = channels / 2
        self.meanOnly = meanOnly
        
        self.pre = MLXNN.Conv1d(inputChannels: halfChannels, outputChannels: hiddenChannels, kernelSize: 1)
        self.enc = WaveNet(hiddenChannels: hiddenChannels, kernelSize: kernelSize, dilationRate: dilationRate, nLayers: nLayers, ginChannels: ginChannels)
        let postOutChannels = meanOnly ? halfChannels : halfChannels * 2
        self.post = MLXNN.Conv1d(inputChannels: hiddenChannels, outputChannels: postOutChannels, kernelSize: 1)
        
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray, xMask: MLXArray, g: MLXArray?, reverse: Bool = false) -> MLXArray {
        // x: (B, T, C)
        let x0 = x[0..., 0..., 0..<halfChannels]
        var x1 = x[0..., 0..., halfChannels...]
        
        var h = pre(x0) * xMask
        h = enc(h, xMask: xMask, g: g)
        let stats = post(h) * xMask
        
        let m: MLXArray
        let logs: MLXArray
        
        if meanOnly {
            m = stats
            logs = MLX.zeros(like: m)
        } else {
            m = stats[0..., 0..., 0..<halfChannels]
            logs = stats[0..., 0..., halfChannels...]
        }
        
        let clampedLogs = MLX.clip(logs, min: -9.0, max: 9.0)
        
        if !reverse {
            x1 = (m + x1 * exp(clampedLogs)) * xMask
        } else {
            x1 = (x1 - m) * exp(-clampedLogs) * xMask
        }
        
        return MLX.concatenated([x0, x1], axis: -1)
    }
}

// MARK: - Residual Coupling Block (flow)

class ResidualCouplingBlock: Module {
    let nFlows: Int

    let flow_0: ResidualCouplingLayer
    let flow_1: ResidualCouplingLayer
    let flow_2: ResidualCouplingLayer
    let flow_3: ResidualCouplingLayer

    init(channels: Int, hiddenChannels: Int, kernelSize: Int, dilationRate: Int, nLayers: Int, nFlows: Int = 4, ginChannels: Int) {
        assert(nFlows == 4, "ResidualCouplingBlock hardcoded for 4 flows")
        self.nFlows = nFlows

        self.flow_0 = ResidualCouplingLayer(channels: channels, hiddenChannels: hiddenChannels, kernelSize: kernelSize, dilationRate: dilationRate, nLayers: nLayers, ginChannels: ginChannels, meanOnly: true)
        self.flow_1 = ResidualCouplingLayer(channels: channels, hiddenChannels: hiddenChannels, kernelSize: kernelSize, dilationRate: dilationRate, nLayers: nLayers, ginChannels: ginChannels, meanOnly: true)
        self.flow_2 = ResidualCouplingLayer(channels: channels, hiddenChannels: hiddenChannels, kernelSize: kernelSize, dilationRate: dilationRate, nLayers: nLayers, ginChannels: ginChannels, meanOnly: true)
        self.flow_3 = ResidualCouplingLayer(channels: channels, hiddenChannels: hiddenChannels, kernelSize: kernelSize, dilationRate: dilationRate, nLayers: nLayers, ginChannels: ginChannels, meanOnly: true)

        super.init()
    }

    func callAsFunction(_ x: MLXArray, xMask: MLXArray, g: MLXArray?, reverse: Bool = false) -> MLXArray {
        var h = x
        let flows = [flow_0, flow_1, flow_2, flow_3]

        if !reverse {
            for i in 0..<nFlows {
                h = flows[i](h, xMask: xMask, g: g, reverse: false)
                h = h[0..., 0..., .stride(by: -1)]
                MLX.eval(h)
            }
        } else {
            for i in (0..<nFlows).reversed() {
                h = h[0..., 0..., .stride(by: -1)]
                MLX.eval(h)
                h = flows[i](h, xMask: xMask, g: g, reverse: true)
            }
        }

        return h
    }
}

// MARK: - Full Synthesizer

public class Synthesizer: Module {
    let enc_p: TextEncoder
    let dec: Generator
    let flow: ResidualCouplingBlock
    let emb_g: MLXNN.Embedding
    let useF0: Bool
    
    public init(
        interChannels: Int = 192,
        hiddenChannels: Int = 192,
        filterChannels: Int = 768,
        nHeads: Int = 2,
        nLayers: Int = 6,
        kernelSize: Int = 3,
        pDropout: Float = 0.0,
        embeddingDim: Int = 768,
        speakerEmbedDim: Int = 256,
        ginChannels: Int = 256,
        useF0: Bool = true,
        upsampleRates: [Int] = [10, 10, 2, 2],
        upsampleKernelSizes: [Int] = [16, 16, 4, 4],
        sampleRate: Int = 40000
    ) {
        self.useF0 = useF0

        self.enc_p = TextEncoder(
            outChannels: interChannels,
            hiddenChannels: hiddenChannels,
            filterChannels: filterChannels,
            nHeads: nHeads,
            nLayers: nLayers,
            kernelSize: kernelSize,
            pDropout: pDropout,
            embeddingDim: embeddingDim,
            f0: useF0
        )

        self.dec = Generator(inputChannels: interChannels, ginChannels: ginChannels, upsampleRates: upsampleRates, upsampleKernelSizes: upsampleKernelSizes, sampleRate: sampleRate)
        
        self.flow = ResidualCouplingBlock(
            channels: interChannels,
            hiddenChannels: hiddenChannels,
            kernelSize: 5,
            dilationRate: 1,
            nLayers: 3,
            nFlows: 4,
            ginChannels: ginChannels
        )
        
        self.emb_g = MLXNN.Embedding(embeddingCount: speakerEmbedDim, dimensions: ginChannels)
        
        super.init()
    }
    
    public func infer(phone: MLXArray, phoneLengths: MLXArray, pitch: MLXArray?, nsff0: MLXArray?, sid: MLXArray) -> MLXArray {
        let g = emb_g(sid).expandedDimensions(axis: 1)
        
        // m_p, logs_p, xMask はすべて (B, T, C) 形式
        let (m_p, logs_p, xMask) = enc_p(phone, pitch: pitch, lengths: phoneLengths)
        
        let clampedLogsP = MLX.clip(logs_p, min: -9.0, max: 9.0)
        let z_p = (m_p + exp(clampedLogsP) * MLXRandom.normal(m_p.shape).asType(m_p.dtype) * 0.0) * xMask

        // Flow 逆変換 (B, T, C)
        let z = flow(z_p, xMask: xMask, g: g, reverse: true)

        // MLX Conv1d 仕様に合わせて転置は行わず、(B, T, C) のまま渡す
        let output = dec(z * xMask, f0: nsff0 ?? MLX.zeros([phone.shape[0], phone.shape[1], 1]), g: g)
        
        return output
    }
}
