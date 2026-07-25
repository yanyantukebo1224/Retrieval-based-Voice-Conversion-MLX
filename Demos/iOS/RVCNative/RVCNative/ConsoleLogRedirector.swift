import Foundation
import Combine

class ConsoleLogRedirector: ObservableObject {
    static let shared = ConsoleLogRedirector()
    
    @Published var logs: String = ""
    private let pipe = Pipe()
    private var originalStdout: Int32 = -1
    private var originalStderr: Int32 = -1
    private var logFileHandle: FileHandle?
    
    private init() {
        setupLogDirectoryAndFile()
        setupRedirect()
    }
    
    private func setupLogDirectoryAndFile() {
        let fileManager = FileManager.default
        guard let docsDir = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
        let logsDir = docsDir.appendingPathComponent("logs", isDirectory: true)
        
        do {
            if !fileManager.fileExists(atPath: logsDir.path) {
                try fileManager.createDirectory(at: logsDir, withIntermediateDirectories: true, attributes: nil)
            }
            let logFileURL = logsDir.appendingPathComponent("rvc_app.log")
            if !fileManager.fileExists(atPath: logFileURL.path) {
                fileManager.createFile(atPath: logFileURL.path, contents: nil, attributes: nil)
            }
            logFileHandle = try FileHandle(forWritingTo: logFileURL)
            logFileHandle?.seekToEndOfFile()
        } catch {
            print("Failed to initialize logs directory/file: \(error)")
        }
    }
    
    func setupRedirect() {
        // Save original descriptors
        originalStdout = dup(STDOUT_FILENO)
        originalStderr = dup(STDERR_FILENO)
        
        // Redirect stdout and stderr to our pipe
        dup2(pipe.fileHandleForWriting.fileDescriptor, STDOUT_FILENO)
        dup2(pipe.fileHandleForWriting.fileDescriptor, STDERR_FILENO)
        
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            
            // 1. Write back to original stdout so it still shows in Xcode console (TEE)
            if let stdout = self?.originalStdout, stdout != -1 {
                data.withUnsafeBytes { ptr in
                    _ = write(stdout, ptr.baseAddress, data.count)
                }
            }
            
            // 2. Write to logs/rvc_app.log file
            if let handle = self?.logFileHandle {
                try? handle.write(contentsOf: data)
            }
            
            // 3. Update the in-app log buffer
            if let str = String(data: data, encoding: .utf8) {
                DispatchQueue.main.async {
                    self?.logs += str
                    
                    // Limit log size to prevent memory issues (last 10000 chars)
                    if (self?.logs.count ?? 0) > 10000 {
                        self?.logs = String((self?.logs.suffix(8000))!)
                    }
                }
            }
        }
    }
    
    func clear() {
        DispatchQueue.main.async {
            self.logs = ""
        }
    }
    
    deinit {
        logFileHandle?.closeFile()
    }
}
