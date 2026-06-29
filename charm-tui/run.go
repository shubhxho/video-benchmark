package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
)

var ansiRe = regexp.MustCompile("\x1b\\[[0-9;]*m")

// findRepoRoot walks up from the cwd to the dir containing pyproject.toml so the
// Python pipeline runs from the project root regardless of where the binary is.
func findRepoRoot() string {
	dir, err := os.Getwd()
	if err != nil {
		return "."
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "pyproject.toml")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "."
		}
		dir = parent
	}
}

// runDistill launches the Python distillation, streams its progress lines, and
// delivers the parsed Result (or an error) over sub.
func runDistill(a runArgs, sub chan tea.Msg) {
	root := findRepoRoot()
	cmd := exec.Command(
		"uv", "run", "python", "-m", "video_benchmark.distill", "--emit-json",
		"--videos", a.videos,
		"--epochs", strconv.Itoa(a.epochs),
		"--fps", strconv.FormatFloat(a.fps, 'g', -1, 64),
	)
	cmd.Dir = root

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		sub <- errMsg{err}
		return
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		sub <- errMsg{err}
		return
	}
	if err := cmd.Start(); err != nil {
		sub <- errMsg{err}
		return
	}

	// Stream the "▸ ..." step lines from stderr as progress messages.
	go func() {
		sc := bufio.NewScanner(stderr)
		sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
		for sc.Scan() {
			line := strings.TrimSpace(ansiRe.ReplaceAllString(sc.Text(), ""))
			if i := strings.Index(line, "▸"); i >= 0 {
				step := strings.TrimSpace(line[i+len("▸"):])
				if step != "" {
					sub <- progressMsg(step)
				}
			}
		}
	}()

	data, _ := io.ReadAll(stdout)
	waitErr := cmd.Wait()

	line := lastJSONLine(data)
	if line == "" {
		if waitErr != nil {
			sub <- errMsg{fmt.Errorf("distillation failed: %w", waitErr)}
		} else {
			sub <- errMsg{fmt.Errorf("no JSON results on stdout")}
		}
		return
	}
	var res Result
	if err := json.Unmarshal([]byte(line), &res); err != nil {
		sub <- errMsg{fmt.Errorf("parsing results: %w", err)}
		return
	}
	sub <- doneMsg{&res}
}

func lastJSONLine(b []byte) string {
	lines := strings.Split(string(b), "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		s := strings.TrimSpace(lines[i])
		if strings.HasPrefix(s, "{") {
			return s
		}
	}
	return ""
}
