import SwiftUI

struct ContentView: View {
    @StateObject private var engine = Engine()
    @State private var prompt = ""
    @State private var showSettings = false

    var body: some View {
        NavigationStack {
            content
                .navigationTitle("BMoE Example")
                .toolbar {
                    ToolbarItem(placement: .topBarTrailing) {
                        Button { showSettings = true } label: { Image(systemName: "slider.horizontal.3") }
                            .disabled(!isIdle)
                    }
                    ToolbarItem(placement: .topBarLeading) {
                        if case .ready = engine.state {
                            Button("Eject") { engine.closeSession() }
                        }
                    }
                }
                .sheet(isPresented: $showSettings) { SettingsView(settings: $engine.settings) }
        }
    }

    private var isIdle: Bool {
        if case .idle = engine.state { return true }
        return false
    }

    @ViewBuilder private var content: some View {
        switch engine.state {
        case .idle:
            ModelPicker { engine.open(modelPath: $0) }
        case .loading:
            VStack(spacing: 12) {
                ProgressView()
                Text("Loading model — tens of seconds for a large one").font(.footnote)
            }
        case .failed(let message):
            VStack(spacing: 12) {
                Text(message).font(.footnote).foregroundStyle(.red)
                Button("Back") { engine.closeSession() }
            }
            .padding()
        case .ready(let model), .generating(let model):
            chat(model: model)
        }
    }

    private func chat(model: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(model).font(.caption).foregroundStyle(.secondary)
            ScrollView {
                VStack(alignment: .leading, spacing: 8) {
                    if !engine.reasoning.isEmpty {
                        DisclosureGroup("Thinking") {
                            Text(engine.reasoning).font(.footnote).foregroundStyle(.secondary)
                        }
                    }
                    Text(engine.answer.isEmpty ? "…" : engine.answer)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
            }
            if case .generating = engine.state {
                Text(engine.liveTelemetry).font(.caption2.monospaced()).foregroundStyle(.secondary)
            } else {
                Text(engine.stats).font(.caption2.monospaced()).foregroundStyle(.secondary)
            }
            HStack {
                TextField("Prompt", text: $prompt, axis: .vertical).textFieldStyle(.roundedBorder)
                if case .generating = engine.state {
                    Button { engine.cancel() } label: { Image(systemName: "stop.circle.fill") }
                } else {
                    Button {
                        let p = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
                        guard !p.isEmpty else { return }
                        engine.generate(prompt: p)
                    } label: { Image(systemName: "arrow.up.circle.fill") }
                }
            }
        }
        .padding()
    }
}

/// Lists *.gguf in Documents. No downloader here: models arrive over Finder / the Files app
/// (UIFileSharingEnabled) — for a multi-shard model copy every shard and pick the first.
private struct ModelPicker: View {
    let onPick: (String) -> Void
    @State private var models: [URL] = []

    var body: some View {
        List {
            Section {
                ForEach(models, id: \.self) { url in
                    Button {
                        onPick(url.path)
                    } label: {
                        VStack(alignment: .leading) {
                            Text(url.lastPathComponent).font(.subheadline)
                            Text(sizeLabel(url)).font(.caption2).foregroundStyle(.secondary)
                        }
                    }
                }
            } header: {
                Text("Models in Documents")
            } footer: {
                Text("Copy .gguf files in via Finder (phone → Files → BMoE Example) or the Files "
                    + "app. For a sharded model copy all shards and tap the first (-00001-of-…).")
            }
        }
        .overlay { if models.isEmpty { Text("No models yet").foregroundStyle(.secondary) } }
        .refreshable { reload() }
        .onAppear { reload() }
    }

    private func reload() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let all = (try? FileManager.default.contentsOfDirectory(at: docs, includingPropertiesForKeys: nil)) ?? []
        models = all.filter { $0.pathExtension == "gguf" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    private func sizeLabel(_ url: URL) -> String {
        let bytes = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        return ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)
    }
}

private struct SettingsView: View {
    @Binding var settings: EngineSettings
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section("Streaming") {
                    Toggle("Stream experts from flash", isOn: $settings.moeStream)
                    Stepper("Expert cache: \(settings.cacheMB) MiB",
                            value: $settings.cacheMB, in: 0 ... 6000, step: 500)
                    Stepper("I/O lanes: \(settings.ioThreads)", value: $settings.ioThreads, in: 1 ... 8)
                    Toggle("Overlap reads with compute", isOn: $settings.overlap)
                    Toggle("Dense weights anonymous", isOn: $settings.denseAnon)
                }
                Section("Generation") {
                    Stepper("Threads: \(settings.nThreads)", value: $settings.nThreads, in: 1 ... 8)
                    Stepper("Max tokens: \(settings.nPredict)", value: $settings.nPredict, in: 32 ... 1024, step: 32)
                    Stepper("Context: \(settings.nCtx)", value: $settings.nCtx, in: 512 ... 8192, step: 512)
                }
                Section {
                    Text("Applied at the next model load. Defaults are the far-past-RAM recipe "
                        + "from the benchmarks; a cache of 0 disables caching entirely.")
                        .font(.footnote)
                }
            }
            .navigationTitle("Settings")
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
        }
    }
}
