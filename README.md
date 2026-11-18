# Practical 1 – Conflation (Stemming)

```python
import re

S = {"the", "is", "a", "an", "of", "and", "in", "to"}
m = {}

for w in re.findall("[a-z]+", open("in.txt").read().lower()):
    if w in S:
        continue

    r = w
    for x in ("ing", "ed", "ly", "s"):
        if w.endswith(x) and len(w) > len(x) + 2:
            r = w[:-len(x)]
            break

    m.setdefault(r, set()).add(w)

open("out.txt", "w").write(
    "\n".join(f"{k} -> {sorted(v)}" for k, v in sorted(m.items()))
)
```

# Practical 2 – Single-Pass Clustering

```python
from math import sqrt
from collections import Counter

D = [
    "cat dog pet",
    "dog pet animal",
    "stock market share",
    "market share price",
    "cat pet animal"
]

n = lambda x: sum(v * v for v in x.values())
cos = lambda x, y: sum(x[k] * y[k] for k in x) / sqrt(n(x) * n(y))

C = []
M = []

for i, d in enumerate(D):
    v = Counter(d.split())
    b = -1
    best = .3

    for j, c in enumerate(C):
        s = cos(v, c)
        if s >= best:
            best, b = s, j

    if b < 0:
        C.append(v)
        M.append([i])
    else:
        C[b].update(v)
        M[b].append(i)

print(M)
```

# Practical 3 – Inverted File

```python
D = [
    "information retrieval system",
    "retrieval of documents",
    "inverted file index",
    "documents index system"
]

I = {}

for i, d in enumerate(D, 1):
    for w in d.split():
        I.setdefault(w, set()).add(i)

print(I)

print(set.intersection(*[I.get(q, set()) for q in input().split()]))
```

# Practical 4 – Simple Web Crawler (Java)

```java
import java.net.*;
import java.util.*;
import java.util.regex.*;

public class P4 {
    public static void main(String[] a) {
        Queue<String> q = new LinkedList<>(List.of(a[0]));
        Set<String> v = new HashSet<>();

        while (!q.isEmpty() && v.size() < 10) {
            String u = q.poll();

            if (!v.add(u))
                continue;

            System.out.println(u);

            try {
                String h = new String(new URL(u).openStream().readAllBytes());

                Matcher m = Pattern.compile("href=\"(http[^\"]+)\"").matcher(h);

                while (m.find())
                    q.add(m.group(1));

            } catch (Exception e) {
            }
        }
    }
}
```

# Practical 5 – Colour Image Features

```python
import sys
import numpy as np
from PIL import Image

a = np.array(Image.open(sys.argv[1]).convert("RGB"), float)

print(a.mean((0, 1)))

print([
    np.histogram(a[..., i], 8, (0, 256))[0]
    for i in range(3)
])

print(abs(np.diff(a.mean(2), axis=1)).mean())
```
