from django.conf import settings
import pandas as pd
from openpyxl import load_workbook
def Excel2Html(filename):
    # import os
    # from xlrd import open_workbook,formatting
    #
    # ext = filename[-4:]
    # print('ext',ext)
    # if ext == '.pdf':
    #     return ''
    # if ext =='.csv':
    #     return ''
    # filepath = os.path.join(settings.MEDIA_ROOT, filename)
    #
    # #
    # # wb = load_workbook(filepath)
    # # sheet = wb.worksheets[0]
    # # print(sheet)
    # # html = '<table class="previewtable" border="1" cellpadding="1" cellspacing="1">'
    #
    # wb = open_workbook(filepath)
    # sheet = wb.sheet_by_index(0)
    # print(sheet)
    # html = '<table class="previewtable" border="1" cellpadding="1" cellspacing="1">'
    #
    # # mergedcells={}
    # # mergedsapn={}
    # # mergedcellvalue={}
    # # for crange in sheet.merged_cells:
    # #     print(crange)
    # #     rlo, rhi, clo, chi = crange
    # #     for rowx in range(rlo, rhi):
    # #         for colx in range(clo, chi):
    # #             print(rlo)
    # #             mergedcells[(rowx,colx)]=False
    # #             value = str(sheet.cell_value(rowx,colx))
    # #             if value.strip() != '':
    # #                 mergedcellvalue[(rlo,clo)]=value
    # #
    # #     mergedcells[(rlo,clo)]=True
    # #     mergedsapn[(rlo,clo)]=(rhi-rlo, chi-clo)
    # #     mergedsapn[(rlo,clo)]=(rhi-rlo, chi-clo)
    # #
    # #
    # # for row in range(sheet.nrows):
    # #     html=html+'<tr>'
    # #     for col in range(sheet.ncols):
    # #         if (row,col) in mergedcells:
    # #             if mergedcells[(row,col)]==True:
    # #                 rspan,cspan = mergedsapn[(row,col)]
    # #                 value = ''
    # #                 if (row,col) in mergedcellvalue:
    # #                     value = mergedcellvalue[(row,col)]
    # #                 html=html+'<td rowspan=%s colspan=%s>%s</td>'  % (rspan, cspan, value)
    # #         else:
    # #             value =sheet.cell_value(row,col)
    # #             html=html+'<td>' + str(value) + '</td>'
    # #
    # #     html=html+'</tr>'
    #
    # nrows=sheet.nrows
    # for row in range(nrows):
    #     html=html+'<tr>'
    #     for cell in sheet.row_values(row):
    #         value = cell
    #         html=html+'<td>' + str(value) + '</td>'
    #     html=html+'</tr>'
    #
    #
    # # for row in sheet.rows:
    # #     html = html + '<tr>'
    # #     for cell in row:
    # #         value = cell.value
    # #         html = html + '<td>' + str(value) + '</td>'
    # #
    # #     html = html + '</tr>'
    #
    # html=html+'</table>'
    #
    # return html

    import os
    res={}
    # file=os.path.join(settings.MEDIA_ROOT, filename)
    file=filename
    df=pd.read_excel(file, usecols=['province', 'city', 'enbid', 'cellid'])

    # print(df.isnull().any())
    df_null=df[df.isnull().values==True]
    if df_null.shape[0]==0:
        res['code']=0
        # html = df.to_html(index=None, classes='previewtable', na_rep='该值不能为空！')
    else:
        # html = df.to_html(index=None, classes='previewtable', na_rep='该值不能为空！')
        res['code'] = 1
        row_null = df_null.index.values.tolist()
        res['row_error']=[i+1 for i in row_null]
        # return html, code,msg
        # raise Exception('存在空行')

    print(res)
    # rows=df.to_dict('records')
    # cols=[{'name':col,'label':col} for col in df.columns]
    # return rows,cols

    # html=df.to_html(index=None,classes='previewtable',na_rep='该值不能为空！')
    # html=html+'''<div class="modal-footer">
    #     <button type="button" class="btn btn-default" data-dismiss="modal">关闭</button>
    #     <button id="save" type="button" class="btn btn-primary">保存</button>
    #   </div>'''
    # print(html)
    res['html'] = df.to_html(index=None, classes='previewtable', na_rep='该值不能为空！')
    return res


