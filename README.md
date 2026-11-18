# Practical 1 – Conflation (Stemming) with File Handling

```java
import java.io.*;
import java.util.*;

public class P1 {
    public static void main(String[] a) throws Exception {
        Set<String> sw = Set.of("the", "is", "a", "an", "of", "and", "in", "to");
        String[] sf = {"ing", "ed", "ly", "s"};
        Map<String, Set<String>> m = new TreeMap<>();

        Scanner s = new Scanner(new File("in.txt"));

        while (s.hasNext()) {
            String w = s.next().toLowerCase().replaceAll("[^a-z]", ""), r = w;

            if (w.isEmpty() || sw.contains(w))
                continue;

            for (String x : sf)
                if (w.endsWith(x) && w.length() > x.length() + 2) {
                    r = w.substring(0, w.length() - x.length());
                    break;
                }

            m.computeIfAbsent(r, k -> new TreeSet<>()).add(w);
        }

        PrintWriter o = new PrintWriter("out.txt");
        m.forEach((k, v) -> o.println(k + " -> " + v));
        o.close();
    }
}
```

# Practical 2 – Single-Pass Clustering

```java
import java.util.*;

public class P2 {
    static double cos(Map<String, Double> x, Map<String, Double> y) {
        double d = 0, p = 0, q = 0;

        for (var e : x.entrySet()) {
            d += e.getValue() * y.getOrDefault(e.getKey(), 0.0);
            p += e.getValue() * e.getValue();
        }

        for (double v : y.values())
            q += v * v;

        return d / Math.sqrt(p * q);
    }

    public static void main(String[] a) {
        String[] D = {
            "cat dog pet",
            "dog pet animal",
            "stock market share",
            "market share price",
            "cat pet animal"
        };

        double T = 0.3;
        List<Map<String, Double>> C = new ArrayList<>();
        List<List<Integer>> M = new ArrayList<>();

        for (int i = 0; i < D.length; i++) {
            Map<String, Double> v = new HashMap<>();

            for (String w : D[i].split(" "))
                v.merge(w, 1.0, Double::sum);

            int b = -1;
            double best = T;

            for (int j = 0; j < C.size(); j++) {
                double s = cos(v, C.get(j));

                if (s >= best) {
                    best = s;
                    b = j;
                }
            }

            if (b < 0) {
                C.add(v);
                M.add(new ArrayList<>(List.of(i)));
            } else {
                for (var e : v.entrySet())
                    C.get(b).merge(e.getKey(), e.getValue(), Double::sum);

                M.get(b).add(i);
            }
        }

        System.out.println(M);
    }
}
```

# Practical 3 – Document Retrieval Using an Inverted File

```java
import java.util.*;

public class P3 {
    public static void main(String[] a) {
        String[] D = {
            "information retrieval system",
            "retrieval of documents",
            "inverted file index",
            "documents index system"
        };

        Map<String, TreeSet<Integer>> I = new TreeMap<>();

        for (int i = 0; i < D.length; i++)
            for (String w : D[i].split(" "))
                I.computeIfAbsent(w, k -> new TreeSet<>()).add(i + 1);

        System.out.println(I);

        Set<Integer> r = null;

        for (String q : new Scanner(System.in).nextLine().split(" ")) {
            Set<Integer> s = I.getOrDefault(q, new TreeSet<>());

            if (r == null)
                r = new TreeSet<>(s);
            else
                r.retainAll(s);
        }

        System.out.println("Docs: " + r);
    }
}
```

# Practical 4 – Simple Web Crawler

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

# Practical 5 – Feature Extraction from a 2D Colour Image

```java
import java.awt.image.*;
import javax.imageio.*;
import java.io.*;
import java.util.*;

public class P5 {
    public static void main(String[] a) throws Exception {
        BufferedImage im = ImageIO.read(new File(a[0]));

        int w = im.getWidth(), h = im.getHeight(), n = w * h;

        double r = 0, g = 0, b = 0, t = 0;
        int[][] H = new int[3][8];

        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                int p = im.getRGB(x, y),
                    R = p >> 16 & 255,
                    G = p >> 8 & 255,
                    B = p & 255;

                r += R;
                g += G;
                b += B;

                H[0][R / 32]++;
                H[1][G / 32]++;
                H[2][B / 32]++;

                if (x > 0) {
                    int q = im.getRGB(x - 1, y);

                    t += Math.abs(
                        (R + G + B) -
                        ((q >> 16 & 255) + (q >> 8 & 255) + (q & 255))
                    ) / 3.0;
                }
            }

        System.out.printf(
            "Mean RGB: %.1f %.1f %.1f%nTexture: %.2f%n",
            r / n, g / n, b / n, t / n
        );

        System.out.println(Arrays.deepToString(H));
    }
}
```
