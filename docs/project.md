---
layout: default
---

# Description

Scientific research is increasingly driven by data-intensive problems. As the complexity of studied
problems is rising, so does their need for high data throughput and capacity. The globally produced
data volume doubles approximately every two years, leading to an exponential data deluge. This
deluge then directly challenges database management systems and file systems, which provide
the foundation for efficient data analysis and management. These systems use different memory
and storage devices, which were traditionally divided into primary, secondary and tertiary memory.
However, with the introduction of the disruptive technology of non-volatile RAM (NVRAM), these
classes started to merge into one another leading to heterogeneous storage architectures, where each
storage device has highly different performance characteristics (e.g., persistence, storage capacity,
latency). Hence, a major challenge is how to exploit the specific characteristics of memory devices.
To this end, SMASH will investigate the benefits of a common storage engine that manages a
heterogeneous storage landscape, including traditional storage devices and non-volatile memory
technologies. The core for this storage engine will be B휀-trees, as they can be used to efficiently exploit
these different devices. Furthermore, data placement and migration strategies will be investigated to
minimize the overhead caused by transferring data between different devices. Eliminating the need
for volatile caches will allow data consistency guarantees to be improved. From the application side,
the storage engine will offer key-value and object interfaces that can be used for a wide range of
use cases, such as high-performance computing and database management systems. Moreover,
due to the widening gap between the performance of computing and storage devices as well as their
stagnating access performance, data reduction techniques are in high demand to reduce the bandwidth
requirements when storing and retrieving data. We will, therefore, conduct research regarding data
transformations in general and the possibilities of external and accelerated transformations. As part of
SMASH, we will provide a prototypical standalone software library to be used by third-party projects.
Common high-performance computing (HPC) workflows will be supported through an integration of
SMASH into the existing JULEA storage framework, while database systems can use the interface of
SMASH directly whenever data is stored or accessed.
The consortium will be supported by Intel Corporation, especially concerning the hardware acceler-
ation of data transformations and collaborate actively with other project within the SPP 2377.
## Header 2

> This is a blockquote following a header.
>
> When something is important enough, you do it even if the odds are not in your favor.

### Header 3

```js
// Javascript code with syntax highlighting.
var fun = function lang(l) {
  dateformat.i18n = require('./lang/' + l)
  return true;
}
```

```ruby
# Ruby code with syntax highlighting
GitHubPages::Dependencies.gems.each do |gem, version|
  s.add_dependency(gem, "= #{version}")
end
```

#### Header 4

*   This is an unordered list following a header.
*   This is an unordered list following a header.
*   This is an unordered list following a header.

##### Header 5

1.  This is an ordered list following a header.
2.  This is an ordered list following a header.
3.  This is an ordered list following a header.

###### Header 6

| head1        | head two          | three |
|:-------------|:------------------|:------|
| ok           | good swedish fish | nice  |
| out of stock | good and plenty   | nice  |
| ok           | good `oreos`      | hmm   |
| ok           | good `zoute` drop | yumm  |

### There's a horizontal rule below this.

* * *

### Here is an unordered list:

*   Item foo
*   Item bar
*   Item baz
*   Item zip

### And an ordered list:

1.  Item one
1.  Item two
1.  Item three
1.  Item four

### And a nested list:

- level 1 item
  - level 2 item
  - level 2 item
    - level 3 item
    - level 3 item
- level 1 item
  - level 2 item
  - level 2 item
  - level 2 item
- level 1 item
  - level 2 item
  - level 2 item
- level 1 item

### Small image

![Octocat](https://github.githubassets.com/images/icons/emoji/octocat.png)

### Large image

![Branching](https://guides.github.com/activities/hello-world/branching.png)


### Definition lists can be used with HTML syntax.

<dl>
<dt>Name</dt>
<dd>Godzilla</dd>
<dt>Born</dt>
<dd>1952</dd>
<dt>Birthplace</dt>
<dd>Japan</dd>
<dt>Color</dt>
<dd>Green</dd>
</dl>

```
Long, single-line code blocks should not wrap. They should horizontally scroll if they are too long. This line should be long enough to demonstrate this.
```

```
The final element.
```
[back](./)
