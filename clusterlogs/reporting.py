from jinja2 import Environment, FileSystemLoader


def generate_html_report(df):
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template("template.html")
    original_list = df.result.to_dict('records')
    new_list = [{k: v for k, v in d.items() if k in ['cluster_size', 'pattern', 'common_phrases_pyTextRank', 'common_phrases_RAKE']} for d in original_list]
    template_vars = {"values": new_list}
    html_out = template.render(template_vars)
    f= open("report.html","w")
    f.write(html_out)
    f.close()