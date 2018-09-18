# -*- coding: utf-8 -*-
from django.conf import settings
from django.forms import Widget
from django.utils.html import format_html
from django.forms.utils import flatatt
from django.utils.encoding import force_text
from django.utils.safestring import mark_safe

class IconWidget(Widget):
    html_template = '''
    <script>
        var icons = %s;
        $(function() {  
            $('#id_%s').fontIconPicker({  
                theme: 'fip-bootstrap',//四种主题风格：fip-grey, fip-darkgrey, fip-bootstrap, fip-inverted
                source:icons,
                //emptyIcon: false,
                iconsPerPage: 105

            });  
        });  
    </script>
    '''

    def __init__(self, attrs=None):
        '''
        为了能在调用的时候自定义代码类型和样式
        :param mode:
        :param theme:
        :param attrs:
        :return:
        '''
        super(IconWidget, self).__init__(attrs)

    def render(self, name, value, attrs=None, renderer=None):
        '''
        关键方法
        :param name:
        :param value:
        :param attrs:
        :return:
        '''
        if value is None:
            value = ''
        self.attrs.update(attrs)
        print(name)
        final_attrs = self.build_attrs(self.attrs, {'name': name})
        output = [format_html('<input{} value="{}">\r\n', flatatt(final_attrs), force_text(value))]
        output.append(self.html_template % (icons, name))
        return mark_safe('\n'.join(output))

    class Media:
        js = (
            'app/js/jquery-1.11.1.min.js',
            'app/lib/highlight/highlight.pack.js'
        )

        css = {
            'all': (
                'app/lib/highlight/styles/dark.css',
            )
        }
