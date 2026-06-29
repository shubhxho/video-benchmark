// charm-tui is a Bubble Tea + Lip Gloss front-end for the Python distillation
// pipeline. It runs `python -m video_benchmark.distill --emit-json`, streams the
// progress live, and renders the results natively with the charm.sh stack.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"strings"

	"github.com/charmbracelet/bubbles/spinner"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

type runArgs struct {
	videos string
	epochs int
	fps    float64
}

type (
	progressMsg string
	doneMsg     struct{ res *Result }
	errMsg      struct{ err error }
)

type model struct {
	sp    spinner.Model
	steps []string
	res   *Result
	err   error
	done  bool
	sub   chan tea.Msg
	args  runArgs
	width int
}

func initialModel(a runArgs) model {
	s := spinner.New()
	s.Spinner = spinner.Dot
	s.Style = lipgloss.NewStyle().Foreground(pink)
	return model{sp: s, sub: make(chan tea.Msg, 16), args: a}
}

func startCmd(a runArgs, sub chan tea.Msg) tea.Cmd {
	return func() tea.Msg {
		go runDistill(a, sub)
		return nil
	}
}

func waitFor(sub chan tea.Msg) tea.Cmd {
	return func() tea.Msg { return <-sub }
}

func (m model) Init() tea.Cmd {
	return tea.Batch(m.sp.Tick, startCmd(m.args, m.sub), waitFor(m.sub))
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
	case tea.KeyMsg:
		switch msg.String() {
		case "q", "ctrl+c", "esc":
			return m, tea.Quit
		}
	case spinner.TickMsg:
		var cmd tea.Cmd
		m.sp, cmd = m.sp.Update(msg)
		return m, cmd
	case progressMsg:
		m.steps = append(m.steps, string(msg))
		return m, waitFor(m.sub)
	case doneMsg:
		m.res = msg.res
		m.done = true
		return m, nil
	case errMsg:
		m.err = msg.err
		m.done = true
		return m, nil
	}
	return m, nil
}

func (m model) View() string {
	if m.err != nil {
		return "\n" + panelStyle.BorderForeground(red).Render(
			lipgloss.NewStyle().Foreground(red).Bold(true).Render("✗ "+m.err.Error()),
		) + "\n"
	}
	if m.res != nil {
		return renderReport(m.res, m.termWidth()) + subStyle.Render("  press q to quit\n")
	}
	return m.runningView()
}

func (m model) termWidth() int {
	if m.width <= 0 {
		return 84
	}
	if m.width > 100 {
		return 100
	}
	return m.width
}

func (m model) runningView() string {
	var b strings.Builder
	b.WriteString("\n")
	title := titleStyle.Render("◆ COMPACT QUALITY MODEL · distillation")
	sub := subStyle.Render(fmt.Sprintf(
		"apple/mobileclip_s0_timm   ·   %.0f fps   ·   %d epochs", m.args.fps, m.args.epochs,
	))
	b.WriteString(bannerStyle.Render(title+"\n"+sub) + "\n")
	for _, s := range m.steps {
		b.WriteString(stepStyle.Render("  ▸ "+s) + "\n")
	}
	b.WriteString("\n  " + m.sp.View() + subStyle.Render(" distilling…") + "\n")
	return b.String()
}

// --- rendering --------------------------------------------------------------

func rule(text string, width int) string {
	label := ruleStyle.Render(" " + text + " ")
	dashes := width - lipgloss.Width(label) - 1
	if dashes < 0 {
		dashes = 0
	}
	return label + faintStyle.Render(strings.Repeat("─", dashes))
}

func fnum(p *float64) string {
	if p == nil || math.IsNaN(*p) {
		return "n/a"
	}
	return fmt.Sprintf("%+.2f", *p)
}

func gauge(p *float64, width int) string {
	if p == nil || math.IsNaN(*p) {
		return faintStyle.Italic(true).Render("n/a")
	}
	v := *p
	frac := v
	if frac < 0 {
		frac = 0
	}
	if frac > 1 {
		frac = 1
	}
	n := int(frac*float64(width) + 0.5)
	c := corrColor(v)
	bar := lipgloss.NewStyle().Foreground(c).Render(strings.Repeat(filled, n)) +
		faintStyle.Render(strings.Repeat(empty, width-n))
	return bar + lipgloss.NewStyle().Foreground(c).Render(fmt.Sprintf("  %+.2f", v))
}

func renderReport(r *Result, width int) string {
	var b strings.Builder
	b.WriteString("\n")

	// banner (with real values now)
	title := titleStyle.Render("◆ COMPACT QUALITY MODEL · distillation")
	sub := subStyle.Render(fmt.Sprintf(
		"%s   ·   %s   ·   %d clips   ·   %.0f fps   ·   %d epochs\nframes %d   train %d   val %d",
		r.Backbone, r.Device, r.Clips, r.Fps, r.Epochs, r.Frames, r.Train, r.Val,
	))
	b.WriteString(bannerStyle.Render(title+"\n"+sub) + "\n\n")

	// headline cards
	compColor := green
	if r.CompositePlcc != nil {
		compColor = corrColor(*r.CompositePlcc)
	}
	deepColor := green
	if r.DeepPlcc != nil {
		deepColor = corrColor(*r.DeepPlcc)
	}
	sizeColor := green
	if !r.Export.Under30mb {
		sizeColor = red
	}
	cards := lipgloss.JoinHorizontal(lipgloss.Top,
		card("COMPOSITE", fnum(r.CompositePlcc), "verdict", compColor),
		card("DEEP SIGNALS", fnum(r.DeepPlcc), "iqa + scene", deepColor),
		card("THROUGHPUT", fmt.Sprintf("%.1f×", r.Speed.SpeedupThroughput),
			fmt.Sprintf("%.0f vs %.0f fps", r.Speed.StudentFps, r.Speed.TeacherFps), green),
		card("SIZE", fmt.Sprintf("%.0f MB", r.Size.Fp16Mb),
			fmt.Sprintf("int8 %.0fMB", r.Size.Int8Mb), sizeColor),
	)
	b.WriteString(cards + "\n\n")

	panel := panelStyle.Width(width)

	// fidelity
	b.WriteString(rule("FIDELITY · student reproduces the teacher", width) + "\n")
	var f strings.Builder
	f.WriteString(subStyle.Render(fmt.Sprintf("%-12s %-6s %7s  %-26s %6s", "signal", "kind", "spread", "agreement (PLCC)", "MAE")) + "\n")
	for _, t := range r.Targets {
		name := t.Name
		if t.Kind == "deep" {
			name = lipgloss.NewStyle().Foreground(violet).Render("◆ " + t.Name)
		} else {
			name = stepStyle.Render("  " + t.Name)
		}
		kind := faintStyle.Render("cv")
		if t.Kind == "deep" {
			kind = badge("deep", violet)
		}
		spread := faintStyle.Render(fmt.Sprintf("%4.0f", t.Std))
		if t.Std < 5 {
			spread = faintStyle.Render("flat")
		}
		maeColor := green
		if t.Mae >= 20 {
			maeColor = red
		} else if t.Mae >= 10 {
			maeColor = amber
		}
		mae := lipgloss.NewStyle().Foreground(maeColor).Render(fmt.Sprintf("%5.1f", t.Mae))
		row := fmt.Sprintf("%-22s %-14s %7s  %-26s %s",
			name, kind, spread, gauge(t.Plcc, 16), mae)
		f.WriteString(row + "\n")
	}
	b.WriteString(panel.Render(strings.TrimRight(f.String(), "\n")) + "\n\n")

	// speed
	b.WriteString(rule("SPEED · the time-complexity win", width) + "\n")
	maxFps := math.Max(r.Speed.StudentFps, r.Speed.TeacherFps)
	b.WriteString(panel.Render(
		fpsBar("student", r.Speed.StudentFps, maxFps, green)+"\n"+
			fpsBar("teacher", r.Speed.TeacherFps, maxFps, amber)+"\n\n"+
			subStyle.Render("one forward pass replaces pyiqa + MobileCLIP scene + CV  →  ")+
			lipgloss.NewStyle().Bold(true).Foreground(green).Render(fmt.Sprintf("▲ %.1f× faster", r.Speed.SpeedupThroughput))+
			faintStyle.Render(fmt.Sprintf("   (%.1f vs %.1f ms/frame)", r.Speed.StudentMs, r.Speed.TeacherMs)),
	) + "\n\n")

	// export
	status := lipgloss.NewStyle().Bold(true).Foreground(green).Render("PASS")
	if !r.Export.Under30mb {
		status = lipgloss.NewStyle().Bold(true).Foreground(red).Render("FAIL")
	}
	b.WriteString(panel.BorderForeground(pink).Render(
		lipgloss.NewStyle().Foreground(green).Render("✓ saved  ")+
			lipgloss.NewStyle().Bold(true).Render(r.Export.Path)+
			subStyle.Render(fmt.Sprintf("   %.1f MB fp16", r.Export.Mb))+"\n"+
			subStyle.Render("under-30MB target  ")+status+
			faintStyle.Render(fmt.Sprintf("     best val loss %.4f", r.Export.BestValLoss)),
	) + "\n")
	return b.String()
}

func fpsBar(label string, fps, maxFps float64, c lipgloss.Color) string {
	width := 22
	n := 0
	if maxFps > 0 {
		n = int(fps/maxFps*float64(width) + 0.5)
	}
	bar := lipgloss.NewStyle().Foreground(c).Render(strings.Repeat(filled, n)) +
		faintStyle.Render(strings.Repeat(empty, width-n))
	return subStyle.Render(fmt.Sprintf("%-9s", label)) + bar +
		lipgloss.NewStyle().Foreground(c).Render(fmt.Sprintf("  %.0f fps", fps))
}

func main() {
	var a runArgs
	var from string
	flag.StringVar(&a.videos, "videos", "videos", "directory of videos")
	flag.IntVar(&a.epochs, "epochs", 300, "training epochs")
	flag.Float64Var(&a.fps, "fps", 3.0, "frames/sec sampled per clip")
	flag.StringVar(&from, "from", "", "render a saved results JSON file instead of running")
	flag.Parse()

	// Non-interactive: render a previously emitted results JSON and exit.
	if from != "" {
		data, err := os.ReadFile(from)
		if err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
			os.Exit(1)
		}
		var res Result
		if err := json.Unmarshal(data, &res); err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
			os.Exit(1)
		}
		fmt.Print(renderReport(&res, 96))
		return
	}

	if _, err := tea.NewProgram(initialModel(a), tea.WithAltScreen()).Run(); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}
