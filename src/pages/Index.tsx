import { useState, useCallback, useEffect, useRef } from "react";
import { 
  Film, 
  Tv, 
  FolderOpen, 
  Music, 
  Play, 
  FileVideo,
  FileAudio,
  Check,
  AlertCircle,
  Download,
  X,
  Settings2,
  History,
  Trash2,
  RotateCcw,
} from "lucide-react";
import { ThemeToggle } from "@/components/ThemeToggle";
import { toast } from "sonner";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

type SyncMode = "movie" | "series";
type ProcessingStatus = "idle" | "processing" | "complete";

interface FileItem {
  id: string;
  name: string;
  path: string;
  type: "video" | "audio";
  size?: number;
}

interface BridgeResult {
  videoFile: string;
  audioFile: string;
  startDelay: number | null;
  endDelay: number | null;
  elapsedMs?: number | null;
  error?: string | null;
}

interface SyncResult extends BridgeResult {
  confidence: "high" | "medium" | "low";
}

interface MediaProbe {
  has_audio: boolean;
  has_video: boolean;
  duration?: number | null;
}

interface HistoryEntry {
  id: string;
  date: Date;
  mode: SyncMode;
  results: SyncResult[];
  fileCount: number;
}

interface PickResponse {
  folder: string | null;
  files: FileItem[];
}

export default function Index() {
  const isTauri = !!(window as unknown as { __TAURI_INTERNALS__?: object }).__TAURI_INTERNALS__;
  const [mode, setMode] = useState<SyncMode>("movie");
  const [videoFiles, setVideoFiles] = useState<FileItem[]>([]);
  const [audioFiles, setAudioFiles] = useState<FileItem[]>([]);
  const [status, setStatus] = useState<ProcessingStatus>("idle");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [showAllHistory, setShowAllHistory] = useState(false);
  const [resultFilter, setResultFilter] = useState<"all" | "high" | "medium" | "low">("all");
  const [segmentDuration, setSegmentDuration] = useState(600);
  const [matchPattern, setMatchPattern] = useState("S(\\d+)E(\\d+)");
  const [results, setResults] = useState<SyncResult[]>([]);
  const [progress, setProgress] = useState({ current: 0, total: 0, percent: 0 });
  const [dragOver, setDragOver] = useState<"video" | "audio" | null>(null);
  const [videoFolder, setVideoFolder] = useState<string | null>(null);
  const [audioFolder, setAudioFolder] = useState<string | null>(null);
  const [videoSource, setVideoSource] = useState<"folder" | "files" | null>(null);
  const [audioSource, setAudioSource] = useState<"folder" | "file" | null>(null);
  const [currentFile, setCurrentFile] = useState<string | null>(null);
  const [fileProgress, setFileProgress] = useState(0);
  const [eta, setEta] = useState<string>("--");
  const [logs, setLogs] = useState<string[]>([]);
  const [showConsole, setShowConsole] = useState(false);
  const processStartRef = useRef<number | null>(null);
  const currentFileRef = useRef<string | null>(null);
  const [probeByPath, setProbeByPath] = useState<Record<string, MediaProbe>>({});
  const [selectedVideoIds, setSelectedVideoIds] = useState<Set<string>>(new Set());
  const [selectedAudioIds, setSelectedAudioIds] = useState<Set<string>>(new Set());
  const [lastVideoFolder, setLastVideoFolder] = useState<string | null>(null);
  const [lastAudioFolder, setLastAudioFolder] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>(() => {
    const saved = localStorage.getItem("syncmaster-history");
    return saved ? JSON.parse(saved) : [];
  });

  // Save history to localStorage
  useEffect(() => {
    localStorage.setItem("syncmaster-history", JSON.stringify(history));
  }, [history]);

  const computeConfidence = useCallback((startDelay: number | null, endDelay: number | null) => {
    if (startDelay === null || endDelay === null) return "low";
    const diff = Math.abs(startDelay - endDelay);
    if (diff < 50) return "high";
    if (diff < 500) return "medium";
    return "low";
  }, []);

  const isOutlier = (startDelay: number | null, endDelay: number | null) => {
    if (startDelay === null || endDelay === null) return true;
    return Math.abs(startDelay - endDelay) > 500;
  };

  const getParentFolder = (filePath: string) => {
    const normalized = filePath.replace(/\\/g, "/");
    const index = normalized.lastIndexOf("/");
    return index > 0 ? normalized.slice(0, index) : null;
  };

  const getUniqueParent = (files: FileItem[]) => {
    const parents = new Set(files.map(file => getParentFolder(file.path)).filter(Boolean));
    if (parents.size === 1) return Array.from(parents)[0] as string;
    return null;
  };

  const formatEta = (ms: number) => {
    const totalSeconds = Math.max(0, Math.round(ms / 1000));
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes}:${seconds.toString().padStart(2, "0")}`;
  };

  const formatSize = (bytes?: number) => {
    if (!bytes || bytes <= 0) return "--";
    const units = ["B", "KB", "MB", "GB"];
    let size = bytes;
    let unitIndex = 0;
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex += 1;
    }
    return `${size.toFixed(unitIndex === 0 ? 0 : 1)}${units[unitIndex]}`;
  };

  const getTotalSize = (files: FileItem[]) =>
    files.reduce((total, file) => total + (file.size || 0), 0);

  useEffect(() => {
    const savedVideo = localStorage.getItem("audiosync-last-video-folder");
    const savedAudio = localStorage.getItem("audiosync-last-audio-folder");
    setLastVideoFolder(savedVideo);
    setLastAudioFolder(savedAudio);
  }, []);

  useEffect(() => {
    if (videoFolder) {
      localStorage.setItem("audiosync-last-video-folder", videoFolder);
      setLastVideoFolder(videoFolder);
    }
  }, [videoFolder]);

  useEffect(() => {
    if (audioFolder) {
      localStorage.setItem("audiosync-last-audio-folder", audioFolder);
      setLastAudioFolder(audioFolder);
    }
  }, [audioFolder]);

  const runProbe = async (file: FileItem) => {
    if (!isTauri) return;
    try {
      const probe = await invoke<MediaProbe>("probe_media", { path: file.path });
      setProbeByPath(prev => ({ ...prev, [file.path]: probe }));
    } catch {
      setProbeByPath(prev => ({ ...prev, [file.path]: { has_audio: false, has_video: false, duration: null } }));
    }
  };

  const getMatchKey = (name: string) => {
    try {
      const regex = new RegExp(matchPattern, "i");
      const match = name.match(regex);
      if (!match || match.length < 3) return null;
      const season = match[1]?.padStart(2, "0");
      const episode = match[2]?.padStart(2, "0");
      if (!season || !episode) return null;
      return `${season}-${episode}`;
    } catch {
      return null;
    }
  };

  const isPatternValid = () => {
    try {
      const regex = new RegExp(matchPattern, "i");
      const match = "S01E02".match(regex);
      return !!(match && match.length >= 3);
    } catch {
      return false;
    }
  };

  const pairingPreview = () => {
    if (mode === "movie") {
      const audioName = audioFiles[0]?.name ?? "No audio selected";
      return videoFiles.map(file => ({
        video: file.name,
        audio: audioName,
        status: audioFiles.length > 0 ? "matched" : "missing",
      }));
    }

    const audioByKey = new Map<string, string>();
    for (const audio of audioFiles) {
      const key = getMatchKey(audio.name);
      if (key && !audioByKey.has(key)) {
        audioByKey.set(key, audio.name);
      }
    }

    return videoFiles.map(file => {
      const key = getMatchKey(file.name);
      const matched = key ? audioByKey.get(key) : null;
      return {
        video: file.name,
        audio: matched ?? "No match",
        status: matched ? "matched" : "missing",
      };
    });
  };

  const handleCopyText = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      toast.success("Copied");
    } catch {
      toast.error("Failed to copy");
    }
  };

  const toggleSelection = (id: string, type: "video" | "audio") => {
    if (type === "video") {
      setSelectedVideoIds(prev => {
        const next = new Set(prev);
        if (next.has(id)) {
          next.delete(id);
        } else {
          next.add(id);
        }
        return next;
      });
    } else {
      setSelectedAudioIds(prev => {
        const next = new Set(prev);
        if (next.has(id)) {
          next.delete(id);
        } else {
          next.add(id);
        }
        return next;
      });
    }
  };

  const handleDrop = useCallback((e: React.DragEvent, type: "video" | "audio") => {
    e.preventDefault();
    setDragOver(null);
    
    const items = Array.from(e.dataTransfer.files);
    const newFiles: FileItem[] = items.map((file, index) => ({
      id: `${type}-${Date.now()}-${index}`,
      name: file.name,
      path: (file as unknown as { path?: string }).path || file.name,
      type,
      size: file.size,
    }));

    if (type === "video") {
      setVideoFiles(prev => [...prev, ...newFiles]);
      setVideoSource("files");
      newFiles.forEach(runProbe);
      setSelectedVideoIds(new Set());
      if (!videoFolder && newFiles.length > 0) {
        const folder = getParentFolder(newFiles[0].path);
        setVideoFolder(folder);
      }
    } else {
      if (mode === "movie") {
        setAudioFiles([newFiles[0]]);
        setAudioSource("file");
        if (newFiles.length > 1) {
          toast.info("Movie mode uses a single audio file. Using the first file.");
        }
      } else {
        setAudioFiles(prev => [...prev, ...newFiles]);
        setAudioSource("folder");
      }
      newFiles.forEach(runProbe);
      setSelectedAudioIds(new Set());
      if (!audioFolder && newFiles.length > 0) {
        const folder = getParentFolder(newFiles[0].path);
        setAudioFolder(folder);
      }
    }
    
    toast.success(`Added ${items.length} ${type} file${items.length > 1 ? 's' : ''}`);
  }, [audioFolder, mode, videoFolder, runProbe]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  const handleSelectFolder = async (type: "video" | "audio") => {
    if (!isTauri) {
      toast.error("File picker is only available in the desktop app.");
      return;
    }
    try {
      if (type === "video") {
        const response = await invoke<PickResponse>("pick_video_files", { mode });
        const mapped = response.files.map((file, index) => ({
          ...file,
          id: `video-${Date.now()}-${index}`,
          type: "video",
        }));
        setVideoFiles(mapped);
        mapped.forEach(runProbe);
        setSelectedVideoIds(new Set());
        setVideoFolder(response.folder);
        setVideoSource(response.folder ? "folder" : null);
        if (response.files.length > 0) {
          toast.success(`Added ${response.files.length} video files`);
        }
      } else {
        const response = await invoke<PickResponse>("pick_audio_files", { mode });
        const mapped = response.files.map((file, index) => ({
          ...file,
          id: `audio-${Date.now()}-${index}`,
          type: "audio",
        }));
        setAudioFiles(mapped);
        mapped.forEach(runProbe);
        setSelectedAudioIds(new Set());
        setAudioFolder(response.folder);
        setAudioSource(mode === "movie" ? "file" : response.folder ? "folder" : null);
        if (response.files.length > 0) {
          toast.success(`Added ${response.files.length} audio file${response.files.length > 1 ? 's' : ''}`);
        }
      }
    } catch (error) {
      toast.error("Failed to open file picker");
    }
  };

  const removeFile = (id: string, type: "video" | "audio") => {
    if (type === "video") {
      setVideoFiles(prev => prev.filter(f => f.id !== id));
      setSelectedVideoIds(prev => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
    } else {
      setAudioFiles(prev => prev.filter(f => f.id !== id));
      setSelectedAudioIds(prev => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
    }
  };

  const removeSelected = (type: "video" | "audio") => {
    if (type === "video") {
      setVideoFiles(prev => prev.filter(file => !selectedVideoIds.has(file.id)));
      setSelectedVideoIds(new Set());
    } else {
      setAudioFiles(prev => prev.filter(file => !selectedAudioIds.has(file.id)));
      setSelectedAudioIds(new Set());
    }
  };

  const handleProcess = async () => {
    if (videoFiles.length === 0 || audioFiles.length === 0) return;
    if (!isTauri) {
      toast.error("Processing is only available in the desktop app.");
      return;
    }

    const derivedVideoFolder = videoFolder || getUniqueParent(videoFiles);
    const derivedAudioFolder = audioFolder || getUniqueParent(audioFiles);

    if (mode === "movie") {
      if (!derivedVideoFolder) {
        toast.error("Select a video folder or drop videos from one folder.");
        return;
      }
      if (audioFiles.length === 0) {
        toast.error("Select an audio file.");
        return;
      }
    } else {
      if (!derivedVideoFolder || !derivedAudioFolder) {
        toast.error("Select both video and audio folders for series mode.");
        return;
      }
    }

    setStatus("processing");
    setProgress({ current: 0, total: videoFiles.length, percent: 0 });
    setResults([]);
    setLogs([]);
    setCurrentFile(null);
    setFileProgress(0);
    currentFileRef.current = null;
    processStartRef.current = Date.now();
    toast.info("Starting analysis...");

    const request = {
      mode,
      video_folder: derivedVideoFolder,
      audio_folder: mode === "series" ? derivedAudioFolder : null,
      audio_file: mode === "movie" ? audioFiles[0]?.path : null,
      video_files: mode === "movie" && videoSource === "files" ? videoFiles.map(file => file.path) : null,
      segment_duration: segmentDuration,
      match_pattern: mode === "series" ? matchPattern : null,
    };

    try {
      const finalResults = await invoke<BridgeResult[]>("start_sync", { request });
      const normalized = finalResults.map(result => ({
        ...result,
        confidence: computeConfidence(result.startDelay, result.endDelay),
      }));
      setResults(normalized);
      setStatus("complete");

      const entry: HistoryEntry = {
        id: `history-${Date.now()}`,
        date: new Date(),
        mode,
        results: normalized,
        fileCount: videoFiles.length,
      };
      setHistory(prev => [entry, ...prev].slice(0, 20));

      toast.success(`Analysis complete! ${normalized.length} files processed.`, {
        description: `${normalized.filter(r => r.confidence === 'high').length} high confidence matches`,
      });
    } catch (error) {
      setStatus("idle");
      processStartRef.current = null;
      const message = error instanceof Error ? error.message : String(error);
      if (message.toLowerCase().includes("canceled")) {
        toast.info("Analysis canceled");
      } else {
        toast.error("Analysis failed. Check logs for details.");
        setLogs(prev => [...prev, `Error: ${message}`].slice(-200));
        setShowConsole(true);
      }
    }
  };

  const handleCancel = async () => {
    if (!isTauri) return;
    try {
      await invoke("cancel_sync");
      toast.info("Canceling current run...");
    } catch {
      toast.error("Failed to cancel");
    }
  };

  const clearAll = (showToast = true) => {
    setVideoFiles([]);
    setAudioFiles([]);
    setResults([]);
    setStatus("idle");
    setProgress({ current: 0, total: 0, percent: 0 });
    setVideoFolder(null);
    setAudioFolder(null);
    setVideoSource(null);
    setAudioSource(null);
    setCurrentFile(null);
    setFileProgress(0);
    setEta("--");
    setLogs([]);
    setProbeByPath({});
    setSelectedVideoIds(new Set());
    setSelectedAudioIds(new Set());
    currentFileRef.current = null;
    processStartRef.current = null;
    if (showToast && (videoFiles.length > 0 || audioFiles.length > 0)) {
      toast.info("Cleared all files");
    }
  };

  const loadFromHistory = (entry: HistoryEntry) => {
    setResults(entry.results);
    setMode(entry.mode);
    setStatus("complete");
    setShowHistory(false);
    toast.success("Loaded results from history");
  };

  const deleteHistoryEntry = (id: string) => {
    setHistory(prev => prev.filter(e => e.id !== id));
    toast.info("Removed from history");
  };

  const clearHistory = () => {
    setHistory([]);
    toast.info("History cleared");
  };

  const exportHistoryAll = () => {
    if (history.length === 0) return;
    const merged = history.flatMap(entry => entry.results);
    exportResults(merged);
  };

  const exportResults = async (resultsToExport: SyncResult[]) => {
    if (!isTauri) {
      toast.error("Export is only available in the desktop app.");
      return;
    }
    try {
      const savedPath = await invoke<string>("export_csv", { results: resultsToExport });
      await invoke("open_output_folder", { path: savedPath });
      toast.success("Exported to CSV");
    } catch (error) {
      toast.error("Export canceled or failed");
    }
  };

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      if (e.key === "Enter" && videoFiles.length > 0 && audioFiles.length > 0 && status !== "processing") {
        e.preventDefault();
        handleProcess();
      }

      if (e.key === "Escape" && status !== "processing") {
        e.preventDefault();
        clearAll(true);
      }

      if (e.key === "h" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setShowHistory(prev => !prev);
        toast.info(showHistory ? "History closed" : "History opened");
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [videoFiles, audioFiles, status, showHistory]);

  useEffect(() => {
    let unlistenProgress: (() => void) | undefined;
    let unlistenResult: (() => void) | undefined;
    let unlistenDone: (() => void) | undefined;
    let unlistenFileStart: (() => void) | undefined;
    let unlistenFileEnd: (() => void) | undefined;
    let unlistenFileProgress: (() => void) | undefined;
    let unlistenLog: (() => void) | undefined;

    const setup = async () => {
      if (!(window as unknown as { __TAURI_INTERNALS__?: object }).__TAURI_INTERNALS__) {
        return;
      }
      unlistenProgress = await listen<{ processed: number; total: number }>("sync-progress", (event) => {
        const percent = event.payload.total > 0 ? Math.round((event.payload.processed / event.payload.total) * 100) : 0;
        setProgress({ current: event.payload.processed, total: event.payload.total, percent });
        if (processStartRef.current && event.payload.processed > 0) {
          const elapsedMs = Date.now() - processStartRef.current;
          const avgMs = elapsedMs / event.payload.processed;
          const remaining = avgMs * (event.payload.total - event.payload.processed);
          setEta(formatEta(remaining));
        }
      });

      unlistenResult = await listen<BridgeResult>("sync-result", (event) => {
        const payload = event.payload;
        setResults(prev => [
          ...prev,
          { ...payload, confidence: computeConfidence(payload.startDelay, payload.endDelay) },
        ]);
      });

      unlistenDone = await listen<BridgeResult[]>("sync-done", (event) => {
        const normalized = event.payload.map(result => ({
          ...result,
          confidence: computeConfidence(result.startDelay, result.endDelay),
        }));
        setResults(normalized);
        setFileProgress(100);
        setEta("--");
        currentFileRef.current = null;
        processStartRef.current = null;
      });

      unlistenFileStart = await listen<{ file: string }>("sync-file-start", (event) => {
        setCurrentFile(event.payload.file);
        currentFileRef.current = event.payload.file;
        setFileProgress(0);
        setLogs(prev => [...prev, `Starting: ${event.payload.file}`].slice(-200));
      });

      unlistenFileEnd = await listen<{ file: string; elapsed_ms: number }>("sync-file-end", (event) => {
        const seconds = (event.payload.elapsed_ms / 1000).toFixed(1);
        setLogs(prev => [...prev, `Finished: ${event.payload.file} (${seconds}s)`].slice(-200));
      });

      unlistenFileProgress = await listen<{ file: string; percent: number }>("sync-file-progress", (event) => {
        if (currentFileRef.current === event.payload.file) {
          setFileProgress(event.payload.percent);
        }
      });

      unlistenLog = await listen<string>("sync-log", (event) => {
        setLogs(prev => [...prev, event.payload].slice(-200));
      });
    };

    setup();

    return () => {
      unlistenProgress?.();
      unlistenResult?.();
      unlistenDone?.();
      unlistenFileStart?.();
      unlistenFileEnd?.();
      unlistenFileProgress?.();
      unlistenLog?.();
    };
  }, [computeConfidence]);

  const formatDate = (date: Date) => {
    const d = new Date(date);
    return d.toLocaleDateString() + " " + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const filteredResults = resultFilter === "all"
    ? results
    : results.filter(result => result.confidence === resultFilter);

  const visibleHistory = showAllHistory ? history : history.slice(0, 10);
  const statusLabel =
    status === "processing" ? "Processing" : status === "complete" ? "Complete" : "Ready";
  const resultCounts = {
    high: results.filter(result => result.confidence === "high").length,
    medium: results.filter(result => result.confidence === "medium").length,
    low: results.filter(result => result.confidence === "low").length,
    error: results.filter(result => !!result.error).length,
  };

  return (
    <div className="h-screen flex flex-col bg-background">
      {/* Header */}
      <header className="h-12 px-4 flex items-center justify-between bg-card border-b border-border">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-md bg-primary flex items-center justify-center">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" className="text-primary-foreground">
              <path d="M9 18V5l12-2v13" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
              <circle cx="6" cy="18" r="3" stroke="currentColor" strokeWidth="2.5"/>
              <circle cx="18" cy="16" r="3" stroke="currentColor" strokeWidth="2.5"/>
            </svg>
          </div>
          <span className="text-sm font-semibold text-foreground">AudioSyncMaster</span>
          <span className="text-[10px] text-muted-foreground bg-secondary px-1.5 py-0.5 rounded">v2.0</span>
        </div>

        <div className="flex items-center gap-1 bg-secondary p-0.5 rounded-md">
          <button
            onClick={() => { setMode("movie"); clearAll(false); }}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition-colors ${
              mode === "movie" ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <Film className="w-3.5 h-3.5" />
            Movies
          </button>
          <button
            onClick={() => { setMode("series"); clearAll(false); }}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition-colors ${
              mode === "series" ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <Tv className="w-3.5 h-3.5" />
            Series
          </button>
        </div>

        <div className="flex items-center gap-2">
          <span className={`text-[10px] px-2 py-1 rounded border ${
            status === "processing"
              ? "bg-warning/10 text-warning border-warning/20"
              : status === "complete"
              ? "bg-success/10 text-success border-success/20"
              : "bg-secondary text-muted-foreground border-border"
          }`}>
            {statusLabel}
          </span>
          <ThemeToggle />
          <button
            onClick={() => setShowHistory(!showHistory)}
            className={`p-1.5 rounded transition-colors ${
              showHistory ? "bg-accent text-accent-foreground" : "text-muted-foreground hover:text-foreground hover:bg-secondary"
            }`}
            title="History (Ctrl+H)"
          >
            <History className="w-4 h-4" />
          </button>
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className={`p-1.5 rounded transition-colors ${
              showAdvanced ? "bg-accent text-accent-foreground" : "text-muted-foreground hover:text-foreground hover:bg-secondary"
            }`}
          >
            <Settings2 className="w-4 h-4" />
          </button>
        </div>
      </header>

      {/* Main */}
      <main className="flex-1 flex overflow-hidden">
        {/* Content Area */}
        <div className="flex-1 p-4 overflow-auto">
          <div className="max-w-3xl mx-auto space-y-4">
            
            {/* Drop Zones */}
            <div className="grid grid-cols-2 gap-3">
              {/* Video Drop Zone */}
              <div
                onDrop={(e) => handleDrop(e, "video")}
                onDragOver={handleDragOver}
                onDragEnter={() => setDragOver("video")}
                onDragLeave={() => setDragOver(null)}
                className={`relative rounded-lg bg-card p-4 transition-all ${
                  dragOver === "video" ? "ring-2 ring-primary bg-accent/20" : ""
                }`}
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <FolderOpen className="w-4 h-4 text-warning" />
                    <div>
                      <span className="text-sm font-medium text-foreground">Video Files</span>
                      <div className="text-[10px] text-muted-foreground">
                        {videoFiles.length} files • {formatSize(getTotalSize(videoFiles))}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {selectedVideoIds.size > 0 && (
                      <button
                        onClick={() => removeSelected("video")}
                        className="text-[10px] text-muted-foreground hover:text-destructive"
                        title="Remove selected"
                      >
                        Remove
                      </button>
                    )}
                    {videoFiles.length > 0 && (
                      <button
                        onClick={() => {
                          setVideoFiles([]);
                          setSelectedVideoIds(new Set());
                        }}
                        className="text-[10px] text-muted-foreground hover:text-destructive"
                        title="Clear video files"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    )}
                    <button
                      onClick={() => handleSelectFolder("video")}
                      className="text-[10px] text-primary hover:underline"
                    >
                      Browse
                    </button>
                  </div>
                </div>
                
                {videoFiles.length === 0 ? (
                  <div className="py-6 text-center">
                    <p className="text-xs text-muted-foreground">Drop video files here • Click Browse</p>
                    {lastVideoFolder && (
                      <p className="text-[10px] text-muted-foreground mt-2">
                        Last used: {lastVideoFolder}
                      </p>
                    )}
                  </div>
                ) : (
                  <div className="space-y-1.5">
                    {videoFiles.map(file => (
                      <div key={file.id} className="flex items-center gap-2 px-2 py-1.5 rounded bg-secondary/50 group">
                        <input
                          type="checkbox"
                          checked={selectedVideoIds.has(file.id)}
                          onChange={() => toggleSelection(file.id, "video")}
                          className="h-3 w-3 accent-primary"
                        />
                        <FileVideo className="w-3.5 h-3.5 text-warning shrink-0" />
                        <span
                          className="text-xs text-foreground flex-1 break-all cursor-pointer select-text"
                          onClick={() => handleCopyText(file.name)}
                          title="Click to copy filename"
                        >
                          {file.name}
                        </span>
                        <span className="text-[10px] text-muted-foreground">{formatSize(file.size)}</span>
                        {probeByPath[file.path] && (
                          <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                            probeByPath[file.path].has_video ? "bg-success/15 text-success" : "bg-destructive/15 text-destructive"
                          }`}>
                            {probeByPath[file.path].has_video ? "Video OK" : "No video stream"}
                          </span>
                        )}
                        <button
                          onClick={() => removeFile(file.id, "video")}
                          className="opacity-0 group-hover:opacity-100 p-0.5 hover:bg-destructive/20 rounded transition-opacity"
                        >
                          <X className="w-3 h-3 text-muted-foreground hover:text-destructive" />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
                {dragOver === "video" && (
                  <div className="absolute inset-0 rounded-lg border border-primary/40 bg-primary/5 flex items-center justify-center text-xs text-primary pointer-events-none">
                    Drop videos to add
                  </div>
                )}
              </div>

              {/* Audio Drop Zone */}
              <div
                onDrop={(e) => handleDrop(e, "audio")}
                onDragOver={handleDragOver}
                onDragEnter={() => setDragOver("audio")}
                onDragLeave={() => setDragOver(null)}
                className={`relative rounded-lg bg-card p-4 transition-all ${
                  dragOver === "audio" ? "ring-2 ring-primary bg-accent/20" : ""
                }`}
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Music className="w-4 h-4 text-success" />
                    <div>
                      <span className="text-sm font-medium text-foreground">
                        {mode === "movie" ? "Audio File" : "Audio Files"}
                      </span>
                      <div className="text-[10px] text-muted-foreground">
                        {audioFiles.length} files • {formatSize(getTotalSize(audioFiles))}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {audioFiles.length > 0 && (
                      <button
                        onClick={() => {
                          setAudioFiles([]);
                          setSelectedAudioIds(new Set());
                        }}
                        className="text-[10px] text-muted-foreground hover:text-destructive"
                        title="Clear audio files"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    )}
                    {selectedAudioIds.size > 0 && (
                      <button
                        onClick={() => removeSelected("audio")}
                        className="text-[10px] text-muted-foreground hover:text-destructive"
                        title="Remove selected"
                      >
                        Remove
                      </button>
                    )}
                    <button
                      onClick={() => handleSelectFolder("audio")}
                      className="text-[10px] text-primary hover:underline"
                    >
                      Browse
                    </button>
                  </div>
                </div>
                
                {audioFiles.length === 0 ? (
                  <div className="py-6 text-center">
                    <p className="text-xs text-muted-foreground">Drop audio files here • Click Browse</p>
                    {lastAudioFolder && (
                      <p className="text-[10px] text-muted-foreground mt-2">
                        Last used: {lastAudioFolder}
                      </p>
                    )}
                  </div>
                ) : (
                  <div className="space-y-1.5">
                    {audioFiles.map(file => (
                      <div key={file.id} className="flex items-center gap-2 px-2 py-1.5 rounded bg-secondary/50 group">
                        <input
                          type="checkbox"
                          checked={selectedAudioIds.has(file.id)}
                          onChange={() => toggleSelection(file.id, "audio")}
                          className="h-3 w-3 accent-primary"
                        />
                        <FileAudio className="w-3.5 h-3.5 text-success shrink-0" />
                        <span
                          className="text-xs text-foreground flex-1 break-all cursor-pointer select-text"
                          onClick={() => handleCopyText(file.name)}
                          title="Click to copy filename"
                        >
                          {file.name}
                        </span>
                        <span className="text-[10px] text-muted-foreground">{formatSize(file.size)}</span>
                        {probeByPath[file.path] && (
                          <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                            probeByPath[file.path].has_audio ? "bg-success/15 text-success" : "bg-destructive/15 text-destructive"
                          }`}>
                            {probeByPath[file.path].has_audio ? "Audio OK" : "No audio stream"}
                          </span>
                        )}
                        <button
                          onClick={() => removeFile(file.id, "audio")}
                          className="opacity-0 group-hover:opacity-100 p-0.5 hover:bg-destructive/20 rounded transition-opacity"
                        >
                          <X className="w-3 h-3 text-muted-foreground hover:text-destructive" />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
                {dragOver === "audio" && (
                  <div className="absolute inset-0 rounded-lg border border-primary/40 bg-primary/5 flex items-center justify-center text-xs text-primary pointer-events-none">
                    Drop audio files to add
                  </div>
                )}
              </div>
            </div>

            {(videoFiles.length > 0 || audioFiles.length > 0) && (
              <div className="rounded-lg bg-card border border-border px-4 py-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-foreground">Pairing Preview</span>
                  {mode === "series" && (
                    <span className="text-[10px] text-muted-foreground">
                      Pattern: {matchPattern}
                    </span>
                  )}
                </div>
                {mode === "series" && !isPatternValid() && (
                  <div className="text-[10px] text-warning mb-2">
                    The match pattern does not appear to capture season/episode groups.
                  </div>
                )}
                <div className="space-y-1.5">
                  {pairingPreview().map((pair, index) => (
                    <div key={`${pair.video}-${index}`} className="flex items-start gap-2 text-xs">
                      <span className="text-foreground break-all flex-1">{pair.video}</span>
                      <span className="text-muted-foreground">→</span>
                      <span
                        className={`break-all flex-1 ${
                          pair.status === "matched" ? "text-muted-foreground" : "text-destructive"
                        }`}
                      >
                        {pair.audio}
                      </span>
                      {pair.status === "missing" && (
                        <span className="text-[10px] text-destructive">Missing</span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Progress Bar */}
            {status === "processing" && (
              <div className="rounded-lg bg-card border border-border px-4 py-3">
                <div className="flex items-center justify-between text-[11px] text-muted-foreground mb-2">
                  <div className="flex items-center gap-2">
                    <span className="text-foreground font-medium">Processing</span>
                    {currentFile && (
                      <span className="truncate max-w-[260px]">{currentFile}</span>
                    )}
                  </div>
                  <div className="flex items-center gap-3">
                    <span>{progress.current}/{progress.total}</span>
                    <span>{eta !== "--" ? `ETA ${eta}` : "ETA --"}</span>
                  </div>
                </div>
                <div className="h-1 bg-secondary rounded-full overflow-hidden mb-2">
                  <div
                    className="h-full bg-primary rounded-full transition-all duration-300"
                    style={{ width: `${progress.percent}%` }}
                  />
                </div>
                <div className="flex items-center justify-between text-[10px] text-muted-foreground">
                  <span>Overall {progress.percent}%</span>
                  <span>Current file {fileProgress}%</span>
                </div>
                <div className="h-1 bg-secondary rounded-full overflow-hidden mt-2">
                  <div
                    className="h-full bg-success rounded-full transition-all duration-300"
                    style={{ width: `${fileProgress}%` }}
                  />
                </div>
              </div>
            )}

            {/* Action Button */}
            <div className="flex items-center gap-3">
              <button
                onClick={handleProcess}
                disabled={videoFiles.length === 0 || audioFiles.length === 0 || status === "processing"}
                className="flex items-center gap-2 bg-primary hover:bg-primary/90 disabled:opacity-40 disabled:cursor-not-allowed text-primary-foreground px-5 py-2 rounded-md text-sm font-medium transition-colors"
              >
                <Play className="w-4 h-4" />
                Start Analysis
              </button>
              {status === "processing" && (
                <button
                  onClick={handleCancel}
                  className="flex items-center gap-2 border border-border text-muted-foreground hover:text-foreground hover:bg-secondary px-4 py-2 rounded-md text-sm font-medium transition-colors"
                >
                  Cancel
                </button>
              )}
              
              {(videoFiles.length > 0 || audioFiles.length > 0) && status !== "processing" && (
                <button
                  onClick={() => clearAll(true)}
                  className="p-1.5 rounded hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors"
                  title="Clear all files"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              )}
              
              {status === "complete" && (
                <div className="flex items-center gap-1.5 text-success text-xs">
                  <Check className="w-3.5 h-3.5" />
                  Complete
                </div>
              )}
              <button
                onClick={() => setShowConsole(prev => !prev)}
                className="text-xs text-muted-foreground hover:text-foreground transition-colors"
              >
                {showConsole ? "Hide Console" : "Show Console"}
              </button>
            </div>

            {/* Results */}
            {results.length > 0 && (
              <div className="rounded-lg bg-card overflow-hidden">
                <div className="px-4 py-3 flex items-center justify-between">
                  <span className="text-sm font-medium text-foreground">Results</span>
                  <div className="flex items-center gap-2">
                    <div className="flex items-center gap-1 bg-secondary rounded-md p-0.5">
                      {(["all", "high", "medium", "low"] as const).map((level) => (
                        <button
                          key={level}
                          onClick={() => setResultFilter(level)}
                          className={`px-2 py-1 text-[10px] rounded ${
                            resultFilter === level
                              ? "bg-card text-foreground shadow-sm"
                              : "text-muted-foreground hover:text-foreground"
                          }`}
                        >
                          {level === "all" ? "All" : level.charAt(0).toUpperCase() + level.slice(1)}
                        </button>
                      ))}
                    </div>
                    <button 
                      onClick={() => exportResults(results)}
                      className="flex items-center gap-1.5 text-[10px] text-muted-foreground hover:text-foreground transition-colors"
                    >
                      <Download className="w-3.5 h-3.5" />
                      Export
                    </button>
                  </div>
                </div>
                <div className="px-4 pb-2 text-[10px] text-muted-foreground flex items-center gap-3">
                  <span>High: {resultCounts.high}</span>
                  <span>Medium: {resultCounts.medium}</span>
                  <span>Low: {resultCounts.low}</span>
                  <span>Errors: {resultCounts.error}</span>
                </div>
                {results.some(result => result.confidence === "low" || result.startDelay === null || result.endDelay === null) && (
                  <div className="px-4 pb-2 text-[11px] text-warning">
                    Some results have low confidence or missing end delay. Consider increasing segment duration and re-running.
                  </div>
                )}
                <table className="w-full text-xs">
                  <thead>
                    <tr className="bg-secondary/50">
                      <th className="text-left font-medium text-muted-foreground px-4 py-2 sticky top-0 bg-secondary/50">Video</th>
                      <th className="text-left font-medium text-muted-foreground px-4 py-2 sticky top-0 bg-secondary/50">Audio</th>
                      <th className="text-right font-medium text-muted-foreground px-4 py-2 sticky top-0 bg-secondary/50">Start</th>
                      <th className="text-right font-medium text-muted-foreground px-4 py-2 sticky top-0 bg-secondary/50">End</th>
                      <th className="text-center font-medium text-muted-foreground px-4 py-2 sticky top-0 bg-secondary/50">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredResults.map((result, index) => (
                      <tr key={index} className="border-t border-border/50">
                        <td className="px-4 py-2.5">
                          <div className="flex items-center gap-2">
                            <FileVideo className="w-3.5 h-3.5 text-warning" />
                            <span
                              className="text-foreground break-all cursor-pointer select-text"
                              onClick={() => handleCopyText(result.videoFile)}
                              title="Click to copy video filename"
                            >
                              {result.videoFile}
                            </span>
                          </div>
                        </td>
                        <td className="px-4 py-2.5">
                          <div className="flex items-center gap-2">
                            <FileAudio className="w-3.5 h-3.5 text-success" />
                            <span
                              className="text-muted-foreground break-all cursor-pointer select-text"
                              onClick={() => handleCopyText(result.audioFile)}
                              title="Click to copy audio filename"
                            >
                              {result.audioFile}
                            </span>
                          </div>
                        </td>
                        <td className="px-4 py-2.5 text-right font-mono text-foreground">
                          <span
                            className="cursor-pointer select-text"
                            onClick={() =>
                              handleCopyText(
                                result.startDelay !== null
                                  ? `${result.startDelay > 0 ? "+" : ""}${result.startDelay.toFixed(1)}ms`
                                  : "--"
                              )
                            }
                            title="Click to copy start delay"
                          >
                            {result.startDelay !== null ? `${result.startDelay > 0 ? "+" : ""}${result.startDelay.toFixed(1)}ms` : "--"}
                          </span>
                        </td>
                        <td className="px-4 py-2.5 text-right font-mono text-foreground">
                          <span
                            className="cursor-pointer select-text"
                            onClick={() =>
                              handleCopyText(
                                result.endDelay !== null
                                  ? `${result.endDelay > 0 ? "+" : ""}${result.endDelay.toFixed(1)}ms`
                                  : "--"
                              )
                            }
                            title="Click to copy end delay"
                          >
                            {result.endDelay !== null ? `${result.endDelay > 0 ? "+" : ""}${result.endDelay.toFixed(1)}ms` : "--"}
                          </span>
                        </td>
                        <td className="px-4 py-2.5 text-center">
                          <div className="flex items-center justify-center gap-2">
                            <span
                              title={`Confidence: ${result.confidence}`}
                              className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium ${
                              result.confidence === "high" 
                                ? "bg-success/15 text-success" 
                                : result.confidence === "medium"
                                ? "bg-warning/15 text-warning"
                                : "bg-destructive/15 text-destructive"
                            }`}
                            >
                              {result.confidence === "high" && <Check className="w-2.5 h-2.5" />}
                              {result.confidence === "low" && <AlertCircle className="w-2.5 h-2.5" />}
                              {result.confidence.charAt(0).toUpperCase() + result.confidence.slice(1)}
                            </span>
                            {isOutlier(result.startDelay, result.endDelay) && (
                              <span className="text-[10px] px-2 py-0.5 rounded bg-warning/15 text-warning">
                                Outlier
                              </span>
                            )}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {showConsole && (
              <div className="rounded-lg bg-card p-3">
                <div className="text-[10px] uppercase tracking-wider text-muted-foreground mb-2">
                  Console
                </div>
                {logs.length === 0 ? (
                  <div className="text-[11px] text-muted-foreground">
                    No logs yet. Start an analysis to see output.
                  </div>
                ) : (
                  <div className="max-h-40 overflow-auto text-[11px] font-mono text-muted-foreground space-y-1">
                    {logs.map((line, index) => (
                      <div key={index} className="whitespace-pre-wrap">{line}</div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* History Panel */}
        {showHistory && (
          <div className="w-80 border-l border-border bg-card/60 flex flex-col shadow-apple-md">
            <div className="p-3 flex items-center justify-between border-b border-border">
              <div>
                <span className="text-sm font-medium text-foreground">History</span>
                <div className="text-[10px] text-muted-foreground">{history.length} entries</div>
              </div>
              <div className="flex items-center gap-2">
                {history.length > 0 && (
                  <button
                    onClick={exportHistoryAll}
                    className="text-[10px] text-muted-foreground hover:text-foreground transition-colors"
                  >
                    Export All
                  </button>
                )}
                {history.length > 0 && (
                  <button
                    onClick={clearHistory}
                    className="text-[10px] text-muted-foreground hover:text-destructive transition-colors"
                  >
                    Clear
                  </button>
                )}
              </div>
            </div>
            <div className="flex-1 overflow-auto">
              {history.length === 0 ? (
                <div className="p-4 text-center">
                  <History className="w-8 h-8 text-muted-foreground/30 mx-auto mb-2" />
                  <p className="text-xs text-muted-foreground">No history yet</p>
                </div>
              ) : (
                <div className="p-2 space-y-1">
                  {visibleHistory.map(entry => (
                    <div
                      key={entry.id}
                      className="p-2.5 rounded-lg bg-secondary/30 hover:bg-secondary/50 transition-colors group"
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-[10px] text-muted-foreground">{formatDate(entry.date)}</span>
                        <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                          entry.mode === "movie" ? "bg-warning/15 text-warning" : "bg-primary/15 text-primary"
                        }`}>
                          {entry.mode === "movie" ? "Movie" : "Series"}
                        </span>
                      </div>
                      <p className="text-xs text-foreground mb-2">
                        {entry.fileCount} files • {entry.results.filter(r => r.confidence === 'high').length} high confidence
                      </p>
                      <div className="flex items-center gap-1">
                        <button
                          onClick={() => loadFromHistory(entry)}
                          className="flex-1 flex items-center justify-center gap-1 text-[10px] text-primary hover:bg-primary/10 py-1 rounded transition-colors"
                        >
                          <RotateCcw className="w-3 h-3" />
                          Load
                        </button>
                        <button
                          onClick={() => exportResults(entry.results)}
                          className="flex-1 flex items-center justify-center gap-1 text-[10px] text-muted-foreground hover:text-foreground hover:bg-secondary py-1 rounded transition-colors"
                        >
                          <Download className="w-3 h-3" />
                          Export
                        </button>
                        <button
                          onClick={() => deleteHistoryEntry(entry.id)}
                          className="p-1 text-muted-foreground hover:text-destructive hover:bg-destructive/10 rounded transition-colors opacity-0 group-hover:opacity-100"
                        >
                          <Trash2 className="w-3 h-3" />
                        </button>
                      </div>
                    </div>
                  ))}
                  {history.length > 10 && (
                    <button
                      onClick={() => setShowAllHistory(prev => !prev)}
                      className="w-full text-[10px] text-muted-foreground hover:text-foreground py-2"
                    >
                      {showAllHistory ? "Show less" : "Show more"}
                    </button>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="h-7 px-4 flex items-center justify-between text-[10px] text-muted-foreground bg-card/50">
        <span>{mode === "movie" ? "Movie" : "Series"} Mode</span>
        <div className="flex items-center gap-3">
          <span className="opacity-60">Enter: Start • Esc: Clear • Ctrl+H: History</span>
          <span>Segment: {segmentDuration}s</span>
        </div>
      </footer>

      {showAdvanced && (
        <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4">
          <div className="w-full max-w-md rounded-lg bg-card border border-border shadow-apple-lg">
            <div className="px-4 py-3 border-b border-border flex items-center justify-between">
              <span className="text-sm font-medium text-foreground">Advanced Settings</span>
              <button
                onClick={() => setShowAdvanced(false)}
                className="p-1 rounded hover:bg-secondary text-muted-foreground hover:text-foreground"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="p-4 space-y-4">
              <div>
                <label className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1 block">
                  Segment Duration (seconds)
                </label>
                <input
                  type="number"
                  value={segmentDuration}
                  onChange={(e) => setSegmentDuration(Number(e.target.value))}
                  className="w-full bg-input rounded px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
                />
              </div>
              {mode === "series" && (
                <div>
                  <label className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1 block">
                    Match Pattern
                  </label>
                  <input
                    type="text"
                    value={matchPattern}
                    onChange={(e) => setMatchPattern(e.target.value)}
                    className="w-full bg-input rounded px-3 py-2 text-sm text-foreground font-mono focus:outline-none focus:ring-1 focus:ring-primary"
                  />
                </div>
              )}
              <div className="flex justify-end gap-2">
                <button
                  onClick={() => setShowAdvanced(false)}
                  className="px-3 py-1.5 text-xs rounded border border-border text-muted-foreground hover:text-foreground hover:bg-secondary"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
