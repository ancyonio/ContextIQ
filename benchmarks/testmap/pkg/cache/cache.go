package cache

type Cache struct{ m map[string]int }

func (c *Cache) Get(k string) int { return c.m[k] }
