import csv
import xml.etree.ElementTree
e = xml.etree.ElementTree.parse('/home/mandeep/PycharmProjects/samsadhni/data/a.xml').getroot()

# for atype in e.findall('lexhead'):
#   no=atype.get('no')
#   #print atype.findall('sense')
# print e[0][5].text

for child in e:
    li=[]
    for ch in child.findall('dentry'):
        #print child.tag,child.attrib,ch.tag,ch.text
      li.append(child.attrib['no'])
      li.append(ch.text.encode('utf-8'))
    for ch in child.findall('sense'):
      li.append(ch.text.encode('utf-8'))
      print li
    with open(r'data.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(li)


      # print ch.tag,ch.attrib,ch.text





# from xml.dom.minidom import parse
# dom = parse("/home/mandeep/PycharmProjects/samsadhni/data/a.xml")
# name = dom.getElementsByTagName('sense')
# i=0
#
# print name[3877].firstChild.nodeValue





