"""
Does all sorts of dark magic in order to build/import c++ bfs
"""
import os
import os.path as osp
import setuptools.sandbox

package_abspath = osp.join(*osp.split(osp.abspath(__file__))[:-1])
if not os.path.exists(osp.join(package_abspath, '_search_hnsw.so')):
    # try build _search_hnsw.so
    workdir = os.getcwd()
    try:
        os.chdir(package_abspath)
        setuptools.sandbox.run_setup(osp.join(package_abspath, 'setup.py'), ['clean', 'build'])
        os.system('cp {}/build/lib*/*.so {}/_search_hnsw.so'.format(package_abspath, package_abspath))
        assert os.path.exists(osp.join(package_abspath, '_search_hnsw.so'))
    finally:
        os.chdir(workdir)

try:
    from . import _search_hnsw as search_hnsw_module
except ImportError:
    from . import _search_hnsw as search_hnsw_module

search_hnsw = search_hnsw_module.find_nearest