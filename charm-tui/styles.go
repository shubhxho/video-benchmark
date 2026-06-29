package main

import "github.com/charmbracelet/lipgloss"

// charm/lipgloss palette — matches the Python distill report.
var (
	pink   = lipgloss.Color("#FF6AC1")
	violet = lipgloss.Color("#B794F6")
	green  = lipgloss.Color("#74E39B")
	amber  = lipgloss.Color("#F6C177")
	red    = lipgloss.Color("#FF6B81")
	muted  = lipgloss.Color("#7A7C90")
	faint  = lipgloss.Color("#55576A")
)

const (
	filled = "█"
	empty  = "░"
)

var (
	bannerStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(pink).
			Padding(1, 3)

	titleStyle = lipgloss.NewStyle().Bold(true).Foreground(pink)
	subStyle   = lipgloss.NewStyle().Foreground(muted)
	faintStyle = lipgloss.NewStyle().Foreground(faint)
	stepStyle  = lipgloss.NewStyle().Foreground(muted)

	ruleStyle = lipgloss.NewStyle().Bold(true).Foreground(violet)

	panelStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(faint).
			Padding(1, 2)
)

func card(label, value, sub string, accent lipgloss.Color) string {
	box := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(accent).
		Padding(1, 2).
		Width(20).
		Height(3).
		MarginRight(1)
	body := lipgloss.NewStyle().Bold(true).Foreground(accent).Render(value) + "\n" +
		subStyle.Render(sub)
	head := lipgloss.NewStyle().Bold(true).Foreground(muted).Render(label)
	return box.Render(head + "\n" + body)
}

// corrColor maps a correlation value to the palette.
func corrColor(v float64) lipgloss.Color {
	switch {
	case v >= 0.8:
		return green
	case v >= 0.5:
		return lipgloss.Color("#67E8F9")
	case v >= 0.3:
		return amber
	default:
		return red
	}
}

func badge(text string, bg lipgloss.Color) string {
	return lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("#000000")).
		Background(bg).Padding(0, 1).Render(text)
}
