import yaml
import os.path

location = os.path.dirname(__file__)

data = open(location+'/elements/groups.yml').read()
element_groups = yaml.safe_load(data)

data = open(location+'/elements/data.yml').read()
elements = yaml.safe_load(data)
