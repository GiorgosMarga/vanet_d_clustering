package utils

func GetFileName(filename string) string {
	for idx := len(filename) - 1; idx >= 0; idx-- {
		if filename[idx] == '.' {
			return filename[:idx]
		}
	}
	return filename
}
