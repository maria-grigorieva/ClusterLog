from jinja2 import Environment, FileSystemLoader
import os


def generate_html_report(df, output_file):
    env = Environment(loader=FileSystemLoader(os.path.join(os.path.dirname(__file__))))
    template = env.get_template("template.html")
    original_list = df.to_dict('records')
    new_list = [{k: v for k, v in d.items() if k in ['cluster_size', 'pattern', 'common_phrases_pyTextRank', 'common_phrases_RAKE']} for d in original_list]
    template_vars = {"values": new_list}
    html_out = template.render(template_vars)
    f= open(output_file,"w")
    f.write(html_out)
    f.close()