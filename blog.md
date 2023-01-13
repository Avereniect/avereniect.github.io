---
layout: blog_index
---
# Blog Index
{% for post in site.posts %}
* [{{ post.title }} - {{ post.date | date: "%-D" }}]({{ post.url }})
{% endfor %}
