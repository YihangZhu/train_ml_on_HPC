# get a free port for running in distributed manner using multiple nodes.
from utils.utils import get_free_port

port = get_free_port()

print(f'Free port found: {port}')
