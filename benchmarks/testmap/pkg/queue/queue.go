package queue

type Queue struct{ items []int }

func (q *Queue) Push(x int) { q.items = append(q.items, x) }
