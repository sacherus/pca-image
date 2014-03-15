import urllib2
import simplejson
import cStringIO
import Image

fetcher = urllib2.build_opener()
searchTerm = 'image'
startIndex = 0
searchUrl = "http://ajax.googleapis.com/ajax/services/search/web?v=1.0&q=" + searchTerm + "&start=" + str(startIndex)
print searchUrl
f = fetcher.open(searchUrl)
a = simplejson.load(f)

print a
imageUrl = a['responseData']['results'][0]['unescapedUrl']
#print urllib2.urlopen(imageUrl).read()

#file = cStringIO.StringIO()
#img = Image.open(file)