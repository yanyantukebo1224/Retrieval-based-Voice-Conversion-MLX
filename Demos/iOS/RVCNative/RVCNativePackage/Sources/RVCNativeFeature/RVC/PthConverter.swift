import Foundation
import MLX
import ZIPFoundation

public enum PthConversionError: Error {
    case zipFailed
    case pickleFailed
    case invalidStateDict
    case storageMissing(String)
    case conversionFailed(String)
}

// @MainActor
public final class PthConverter: Sendable {
    public static let shared = PthConverter()
    
    /// Converts a .pth file (PyTorch zip archive) to an MLX-compatible Dictionary of arrays.
    /// - Parameter url: URL to the .pth file
    /// - Parameter copyIndexTo: Optional directory to copy any found .index files to.
    /// - Parameter progress: Closure to report progress (0.0 to 1.0) and status message.
    /// - Returns: A dictionary of [String: MLXArray] ready to be saved or used.
    public func convert(url: URL, copyIndexTo: URL? = nil, progress: (@Sendable (Double, String) -> Void)? = nil) throws -> [String: MLXArray] {
        // 1. Unzip to temp
        progress?(0.05, "Extracting archive...")
        let fm = FileManager.default
        let tempDir = fm.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try fm.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: tempDir) }
        
        try fm.unzipItem(at: url, to: tempDir)
        
        // 2. Inspect Archive & Locate data.pkl
        progress?(0.1, "Inspecting archive structure...")
        
        var dataPklUrl: URL?
        var storageRoot: URL = tempDir
        var modelFiles: [URL] = []
        var indexFiles: [URL] = []
        
        // Log all files for transparency
        if let enumerator = fm.enumerator(at: tempDir, includingPropertiesForKeys: [.isRegularFileKey]) {
            print("--- Archive Structure ---")
            for case let fileURL as URL in enumerator {
                let relPath = fileURL.path.replacingOccurrences(of: tempDir.path, with: "")
                print("  [File] \(relPath)")
                
                if fileURL.lastPathComponent == "data.pkl" {
                    modelFiles.append(fileURL)
                } else if fileURL.pathExtension.lowercased() == "pth" {
                    modelFiles.append(fileURL)
                } else if fileURL.pathExtension.lowercased() == "index" {
                    indexFiles.append(fileURL)
                }
            }
            print("-------------------------")
        }
        
        // Copy index files if requested
        if let destDir = copyIndexTo, !indexFiles.isEmpty {
            for indexFile in indexFiles {
                let dest = destDir.appendingPathComponent(indexFile.lastPathComponent)
                do {
                    if fm.fileExists(atPath: dest.path) {
                        try fm.removeItem(at: dest)
                    }
                    try fm.copyItem(at: indexFile, to: dest)
                    progress?(0.12, "Copied index file: \(indexFile.lastPathComponent)")
                    print("PthConverter: Copied index to \(dest.path)")
                } catch {
                    print("PthConverter: Failed to copy index file: \(error)")
                }
            }
        }
        
        // Helper to find data.pkl in a model URL (might be a directory or a .pth zip)
        func findDataPkl(in root: URL) -> (URL, URL)? {
            if root.lastPathComponent == "data.pkl" {
                return (root, root.deletingLastPathComponent())
            }
            
            // If it's a .pth file, it's actually a zip. Extract it!
            if root.pathExtension.lowercased() == "pth" {
                let nestedTemp = root.deletingLastPathComponent().appendingPathComponent("extracted_\(root.lastPathComponent)")
                do {
                    try fm.createDirectory(at: nestedTemp, withIntermediateDirectories: true)
                    try fm.unzipItem(at: root, to: nestedTemp)
                    
                    if let nestedEnum = fm.enumerator(at: nestedTemp, includingPropertiesForKeys: nil) {
                        for case let f as URL in nestedEnum {
                            if f.lastPathComponent == "data.pkl" {
                                return (f, f.deletingLastPathComponent())
                            }
                        }
                    }
                } catch {
                    print("Failed to extract nested .pth: \(error)")
                }
            }
            return nil
        }
        
        // Pick the best candidate
        if let first = modelFiles.first {
            if modelFiles.count > 1 {
                print("Warning: Multiple model candidates found. Picking: \(first.lastPathComponent)")
            }
            
            if let found = findDataPkl(in: first) {
                dataPklUrl = found.0
                storageRoot = found.1
                progress?(0.15, "Using model: \(first.lastPathComponent)")
            }
        }
        
        guard let pklUrl = dataPklUrl else {
            throw PthConversionError.conversionFailed("No valid model (.pth or data.pkl) found in archive.")
        }
        
        // 3. Unpickle
        progress?(0.2, "Parsing Pickle structure...")
        let data = try Data(contentsOf: pklUrl)
        let unpickler = PickleUnpickler(data: data)
        let unpickledObj = (try? unpickler.load()) ?? [:]
        
        var stateDict: [String: Any] = [:]
        print("PthConverter: Unpickled Type: \(type(of: unpickledObj))")
        if let dict = unpickledObj as? [String: Any] {
            print("PthConverter: Cast to [String: Any] succeeded, count: \(dict.count)")
            stateDict = dict
        } else if let nsDict = unpickledObj as? NSDictionary {
             print("PthConverter: Cast to NSDictionary succeeded, count: \(nsDict.count)")
             for (k, v) in nsDict {
                 if let keyStr = k as? String {
                     stateDict[keyStr] = v
                 }
             }
        } else if let arr = unpickledObj as? [Any] {
             print("PthConverter: Unpickled array count: \(arr.count)")
             for item in arr {
                 if let d = item as? [String: Any] {
                     for (k, v) in d { stateDict[k] = v }
                 } else if let d = item as? NSDictionary {
                     for (k, v) in d { if let ks = k as? String { stateDict[ks] = v } }
                 }
             }
        }
        
        // Check for nested state dict (checkpoint)
        if let nested = stateDict["weight"] as? [String: Any] {
            print("Found 'weight' key, using inner dict.")
            stateDict = nested
        } else if let nested = stateDict["weight"] as? NSDictionary {
             print("Found 'weight' key (NSDictionary), using inner dict.")
             var newDict: [String: Any] = [:]
             for (k, v) in nested {
                 if let ks = k as? String { newDict[ks] = v }
             }
             stateDict = newDict
        } else if let model = stateDict["model"] as? [String: Any] {
            print("Found 'model' key, using inner dict.")
            stateDict = model
        } else if let sd = stateDict["state_dict"] as? [String: Any] {
            print("Found 'state_dict' key, using inner dict.")
            stateDict = sd
        }
        
        // 5. Reconstruct Tensors
        let storageBase = storageRoot
        var mlxDict: [String: MLXArray] = [:]
        
        let totalItems = stateDict.count
        var currentItem = 0
        
        for (key, value) in stateDict {
            autoreleasepool {
                if let tensorRef = value as? TensorReference {
                    if let array = try? loadTensor(ref: tensorRef, baseDir: storageBase) {
                        mlxDict[key] = array
                    }
                }
            }
            
            currentItem += 1
            if currentItem % 5 == 0 {
                let p = 0.2 + (0.7 * Double(currentItem) / Double(totalItems))
                progress?(p, "Loading tensors (\(Int(p * 100))%)...")
            }
        }
        
        // 6. Apply Model Conversion Logic (Weight Norm & Key Remapping)
        progress?(0.9, "Optimizing weights...")
        mlxDict = try processWeights(mlxDict)
        
        progress?(1.0, "Conversion complete!")
        return mlxDict
    }
    
    private func loadTensor(ref: TensorReference, baseDir: URL) throws -> MLXArray {
        let filename = ref.storage.filename
        let storageUrl = baseDir.appendingPathComponent("data").appendingPathComponent(filename)
        
        guard FileManager.default.fileExists(atPath: storageUrl.path) else {
            throw PthConversionError.storageMissing(filename)
        }
        
        let rawData = try Data(contentsOf: storageUrl)
        
        if ref.storage.dtype.contains("FloatStorage") {
            let floats = rawData.withUnsafeBytes {
                Array($0.bindMemory(to: Float.self))
            }
            var array = MLXArray(floats)
            if !ref.size.isEmpty {
                 array = array.reshaped(ref.size)
            }
            return array
        } else if ref.storage.dtype.contains("HalfStorage") {
             let halves = rawData.withUnsafeBytes {
                 Array($0.bindMemory(to: Float16.self))
             }
             var array = MLXArray(halves)
             if !ref.size.isEmpty { array = array.reshaped(ref.size) }
             return array
        } else if ref.storage.dtype.contains("LongStorage") {
             let ints = rawData.withUnsafeBytes {
                 Array($0.bindMemory(to: Int64.self))
             }
             var array = MLXArray(ints)
             if !ref.size.isEmpty { array = array.reshaped(ref.size) }
             return array
        }
        
        return MLXArray(0)
    }
    
    private func processWeights(_ dict: [String: MLXArray]) throws -> [String: MLXArray] {
        var newDict: [String: MLXArray] = [:]
        let keys = Array(dict.keys)
        var processedPrefixes = Set<String>()
        
        // Weight Norm Fusion for ALL layers (conv_pre, conv_post, noise_convs, up, resblocks, etc.)
        for k in keys {
            if k.hasSuffix(".weight_g") {
                let prefix = String(k.dropLast(9))
                if processedPrefixes.contains(prefix) { continue }
                
                guard let w_g = dict[k],
                      let w_v = dict[prefix + ".weight_v"] else {
                    continue
                }
                
                var norm_v: MLXArray
                if w_v.ndim == 3 {
                    norm_v = MLX.sqrt(MLX.sum(w_v * w_v, axes: [1, 2], keepDims: true))
                } else if w_v.ndim == 4 {
                    norm_v = MLX.sqrt(MLX.sum(w_v * w_v, axes: [1, 2, 3], keepDims: true))
                } else {
                    norm_v = MLX.sqrt(MLX.sum(w_v * w_v, axes: [1], keepDims: true))
                }
                
                let w = w_g * (w_v / (norm_v + 1e-12))
                newDict[prefix + ".weight"] = w
                processedPrefixes.insert(prefix)
                
            } else if k.hasSuffix(".weight_v") {
                // Skip
            } else {
                newDict[k] = dict[k]
            }
        }
        
        // Key remapping
        var remappedDict: [String: MLXArray] = [:]
        for (k, v) in newDict {
            var newKey = k

            if newKey.contains("flow.flows.") {
                newKey = newKey.replacingOccurrences(of: "flow.flows.0.", with: "flow.flow_0.")
                newKey = newKey.replacingOccurrences(of: "flow.flows.2.", with: "flow.flow_1.")
                newKey = newKey.replacingOccurrences(of: "flow.flows.4.", with: "flow.flow_2.")
                newKey = newKey.replacingOccurrences(of: "flow.flows.6.", with: "flow.flow_3.")
            }

            let layerPattern = try? NSRegularExpression(pattern: "(in_layers|res_skip_layers)\\.(\\d+)")
            if let regex = layerPattern {
                let range = NSRange(newKey.startIndex..., in: newKey)
                newKey = regex.stringByReplacingMatches(in: newKey, range: range, withTemplate: "$1_$2")
                newKey = newKey.replacingOccurrences(of: "in_layers_", with: "in_layer_")
                newKey = newKey.replacingOccurrences(of: "res_skip_layers_", with: "res_skip_layer_")
            }

            if let noisePattern = try? NSRegularExpression(pattern: "dec\\.noise_convs\\.(\\d+)") {
                let range = NSRange(newKey.startIndex..., in: newKey)
                newKey = noisePattern.stringByReplacingMatches(in: newKey, range: range, withTemplate: "dec.noise_conv_$1")
            }

            if let upsPattern = try? NSRegularExpression(pattern: "dec\\.ups\\.(\\d+)") {
                let range = NSRange(newKey.startIndex..., in: newKey)
                newKey = upsPattern.stringByReplacingMatches(in: newKey, range: range, withTemplate: "dec.up_$1")
            }

            if let resblock1Pattern = try? NSRegularExpression(pattern: "dec\\.resblocks\\.(\\d+)\\.convs1\\.(\\d+)") {
                let range = NSRange(newKey.startIndex..., in: newKey)
                newKey = resblock1Pattern.stringByReplacingMatches(in: newKey, range: range, withTemplate: "dec.resblock_$1.c1_$2")
            }

            if let resblock2Pattern = try? NSRegularExpression(pattern: "dec\\.resblocks\\.(\\d+)\\.convs2\\.(\\d+)") {
                let range = NSRange(newKey.startIndex..., in: newKey)
                newKey = resblock2Pattern.stringByReplacingMatches(in: newKey, range: range, withTemplate: "dec.resblock_$1.c2_$2")
            }

            if let attnPattern = try? NSRegularExpression(pattern: "enc_p\\.encoder\\.attn_layers\\.(\\d+)") {
                let range = NSRange(newKey.startIndex..., in: newKey)
                newKey = attnPattern.stringByReplacingMatches(in: newKey, range: range, withTemplate: "enc_p.encoder.attn_$1")
            }

            if let norm1Pattern = try? NSRegularExpression(pattern: "enc_p\\.encoder\\.norm_layers_1\\.(\\d+)") {
                let range = NSRange(newKey.startIndex..., in: newKey)
                newKey = norm1Pattern.stringByReplacingMatches(in: newKey, range: range, withTemplate: "enc_p.encoder.norm1_$1")
            }

            if let norm2Pattern = try? NSRegularExpression(pattern: "enc_p\\.encoder\\.norm_layers_2\\.(\\d+)") {
                let range = NSRange(newKey.startIndex..., in: newKey)
                newKey = norm2Pattern.stringByReplacingMatches(in: newKey, range: range, withTemplate: "enc_p.encoder.norm2_$1")
            }

            if let ffnPattern = try? NSRegularExpression(pattern: "enc_p\\.encoder\\.ffn_layers\\.(\\d+)") {
                let range = NSRange(newKey.startIndex..., in: newKey)
                newKey = ffnPattern.stringByReplacingMatches(in: newKey, range: range, withTemplate: "enc_p.encoder.ffn_$1")
            }

            if newKey.hasSuffix(".gamma") {
                newKey = String(newKey.dropLast(6)) + ".weight"
            } else if newKey.hasSuffix(".beta") {
                newKey = String(newKey.dropLast(5)) + ".bias"
            }

            var val = v
            if val.ndim == 3 && newKey.hasSuffix(".weight") {
                if newKey.contains("dec.up_") || newKey.contains("dec.ups.") {
                    val = val.transposed(axes: [1, 2, 0])
                } else {
                    val = val.transposed(axes: [0, 2, 1])
                }
            }

            remappedDict[newKey] = val
        }
        
        return remappedDict
    }
}
