#!/usr/bin/python
# -*- coding: UTF-8 -*-

import requests
import sys

if sys.version > '3':
    import urllib.request as urllib2  
else: 
    import urllib2


def wx_get_req(url, params):
    return requests.get(url, params)

def wx_post_req(url, params, img=None, bodyStr=None):
    ret = file_get_contents(img) if img else bodyStr
    return requests.post(url, params=params, data=ret)

#img to str
def file_get_contents(filename, use_include_path = 0, context = None, offset = -1, maxlen = -1):
    if (filename.find('://') > 0):
        ret = urllib2.urlopen(filename).read()
        if (offset > 0):
            ret = ret[offset:]
        if (maxlen > 0):
            ret = ret[:maxlen]
        return ret
    else:
        fp = open(filename,'rb')
        try:
            if (offset > 0):
                fp.seek(offset)
            ret = fp.read(maxlen)
            return ret
        finally:
            fp.close( )
