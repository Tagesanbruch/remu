// Generic ring buffer for trace storage

pub struct RingBuffer<T> {
    buffer: Vec<T>,
    head: usize,
    tail: usize,
    size: usize,
    capacity: usize,
}

impl<T: Clone> RingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            head: 0,
            tail: 0,
            size: 0,
            capacity,
        }
    }

    pub fn push(&mut self, item: T) {
        if self.capacity == 0 {
            return;
        }

        if self.size < self.capacity {
            self.buffer.push(item);
            self.size += 1;
        } else {
            self.buffer[self.tail] = item;
            self.tail = (self.tail + 1) % self.capacity;
            self.head = (self.head + 1) % self.capacity;
        }
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn iter(&self) -> RingBufferIter<T> {
        RingBufferIter {
            buffer: self,
            index: 0,
        }
    }
}

pub struct RingBufferIter<'a, T> {
    buffer: &'a RingBuffer<T>,
    index: usize,
}

impl<'a, T: Clone> Iterator for RingBufferIter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.buffer.size {
            return None;
        }

        let pos = (self.buffer.head + self.index) % self.buffer.capacity.max(1);
        self.index += 1;
        
        if pos < self.buffer.buffer.len() {
            Some(self.buffer.buffer[pos].clone())
        } else {
            None
        }
    }
}
