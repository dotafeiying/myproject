from collections import defaultdict
from ..models import SqlModel

def generate_sql(name,dateStart,dateEnd,enbList,limit, offset,**kwargs):
    result = defaultdict(list)
    for type, value in enbList:
        result[type].append(value)
    if result.get('province') == ['湖北省']:
        citys = '[select DISTINCT city from busycell]'
    else:
        citys = result.get('city', ' "" ')
    cells = result.get('cell', ' "" ')

    count_str="select count(*) as total from busycell "
    query_str="select * from busycell "
    condition="where (enbid like '{0}%' or cellname like '{0}%') " \
                "and unix_timestamp(finish_time) > unix_timestamp('{1}') and unix_timestamp(finish_time) < unix_timestamp('{2}')" \
                "and (city in ({3}) or concat(enbid,'_',cellid) in ({4})) " \
                "".format(name, dateStart, dateEnd, str(citys)[1:-1], str(cells)[1:-1])
    for condition_word,value in kwargs.items():
        if value=="全部":
            pass
        elif value:
            condition=condition+" and {0}='{1}'".format(condition_word,value)

    count_sql = count_str + condition
    query_sql = query_str + condition + " order by finish_time limit {0} offset {1}".format(limit, offset)
    export_sql = query_str + condition + " order by finish_time "

    sqlmodel=SqlModel()
    sqlmodel.count_sql=count_sql
    sqlmodel.query_sql = query_sql
    sqlmodel.export_sql = export_sql
    sqlmodel.save()
    return sqlmodel.pk
