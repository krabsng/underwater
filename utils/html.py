import dominate
from dominate.tags import meta, h3, table, tr, td, p, a, img, br
import os


class HTML:
    """这个 HTML 类允许我们将图像和文本写入单个 HTML 文件。

     它由诸如<add_header>（向HTML文件添加文本标题）等功能组成。
     <add_images> （将一行图像添加到 HTML 文件），然后<save>（将 HTML 保存到磁盘）。
     它基于 Python 库“dominate”，这是一个用于使用 DOM API 创建和操作 HTML 文档的 Python 库。
    """

    def __init__(self, web_dir, title, refresh=0):
        """初始化 HTML 类

        Parameters:
            web_dir (str) -- 存储网页的目录。HTML 文件将在 <web_dir>/index.html 处创建; 图片将保存在 <web_dir/images/
            title (str)   -- 网页名称
            refresh (int) -- 网站多久刷新一次;如果为 0;没有刷新
        """
        self.title = title
        self.web_dir = web_dir
        self.img_dir = os.path.join(self.web_dir, 'images')
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        self.doc = dominate.document(title=title)
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(refresh))

    def get_image_dir(self):
        """返回存储图像的目录"""
        return self.img_dir

    def add_header(self, text):
        """向 HTML 文件插入header

        Parameters:
            text (str) -- the header text
        """
        with self.doc:
            h3(text)

    def add_images(self, ims, txts, links, width=400):
        """向 HTML 文件添加图像

        Parameters:
            ims (str list)   -- 图像路径列表
            txts (str list)  -- 网站上显示的图片名称列表
            links (str list) --  Hyperref 链接列表;当您单击图像时，它会将您重定向到一个新页面
        """
        self.t = table(border=1, style="table-layout: fixed;")  # 插入表格
        self.doc.add(self.t)
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=os.path.join('images', link)):
                                img(style="width:%dpx" % width, src=os.path.join('images', im))
                            br()
                            p(txt)

    def save(self):
        """将当前内容保存到 HMTL 文件中"""
        html_file = '%s/index.html' % self.web_dir
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()


if __name__ == '__main__':  # 这里展示了一个示例用法。
    html = HTML('web/', 'test_html')
    html.add_header('hello world')

    ims, txts, links = [], [], []
    for n in range(4):
        ims.append('image_%d.png' % n)
        txts.append('text_%d' % n)
        links.append('image_%d.png' % n)
    html.add_images(ims, txts, links)
    html.save()
