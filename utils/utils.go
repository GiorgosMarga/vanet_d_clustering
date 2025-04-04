package utils

import (
	"path/filepath"
	"runtime"
)

func GetFileName(filename string) string {
	for idx := len(filename) - 1; idx >= 0; idx-- {
		if filename[idx] == '.' {
			return filename[:idx]
		}
	}
	return filename
}

func GetProjectRoot() string {
	_, filename, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(filename), "..")
}
