package main

// Result mirrors the JSON emitted by `python -m video_benchmark.distill --emit-json`.
// Correlations that are statistically meaningless (near-constant teacher signal)
// arrive as JSON null, so they are pointers — nil renders as "n/a".

type Target struct {
	Name string   `json:"name"`
	Kind string   `json:"kind"` // "deep" (distilled) or "cv" (exact)
	Std  float64  `json:"std"`
	Plcc *float64 `json:"plcc"`
	Srcc *float64 `json:"srcc"`
	Mae  float64  `json:"mae"`
}

type Size struct {
	ParamsMillions float64 `json:"params_millions"`
	Fp16Mb         float64 `json:"fp16_mb"`
	Int8Mb         float64 `json:"int8_mb"`
}

type Speed struct {
	StudentFps        float64 `json:"student_fps"`
	TeacherFps        float64 `json:"teacher_fps"`
	StudentMs         float64 `json:"student_ms"`
	TeacherMs         float64 `json:"teacher_ms"`
	SpeedupThroughput float64 `json:"speedup_throughput"`
	SpeedupLatency    float64 `json:"speedup_latency"`
}

type Export struct {
	Path        string  `json:"path"`
	Mb          float64 `json:"mb"`
	Under30mb   bool    `json:"under_30mb"`
	BestValLoss float64 `json:"best_val_loss"`
}

type Result struct {
	Backbone      string   `json:"backbone"`
	Device        string   `json:"device"`
	Clips         int      `json:"clips"`
	Fps           float64  `json:"fps"`
	Epochs        int      `json:"epochs"`
	Frames        int      `json:"frames"`
	Train         int      `json:"train"`
	Val           int      `json:"val"`
	Targets       []Target `json:"targets"`
	CompositePlcc *float64 `json:"composite_plcc"`
	CompositeSrcc *float64 `json:"composite_srcc"`
	DeepPlcc      *float64 `json:"deep_plcc"`
	Size          Size     `json:"size"`
	Speed         Speed    `json:"speed"`
	Export        Export   `json:"export"`
}
