import requests
"""
response = requests.get("http://www.cmu.edu")

print("Status code: ", response.status_code)
print("Headers: ", response.headers)
print(response.content[:100])
print(response.text[:480])
print("/n")
"""
r = requests.get("https://api.github.com/events")
print(r.text[:100])
print('\n')

params = {"query":"python download url content","source":"chrome"}
response = requests.get("http://www.google.com/search", params=params)
print(response.text[:200])
print("\n")

tokeen = "d7e9d1af2775deb153839aabc9f8c7042ab445c3"
response = requests.get("https://api.github.com/user", params={"access_token":tokeen})
print(response.status_code)
print(response.headers["Content-Type"])
print(response.json())
print(response.json().keys())
print("\n")

# response= requests.get("https://api.github.com/user",auth=("thenatzzz@gmail.com","-----"))
print(response.status_code)
print(response.content)
print("\n--------------------------------------\n")
import json
print(json.loads(response.content))
print("\n")

data = {"a":[1,2,3,{"b":2.3331}],"c":4}
d2 = json.dumps(data)
print(d2)
# print(json.dumps(response))

from bs4 import BeautifulSoup
root = BeautifulSoup("""<tag attribute="value">\
                            <subtag>\
                                content for subtag\
                            </subtag>\
                            <openclosetag attribute="value2"/>\
                            <subtag>\
                             second one\
                            </subtag>\
                        </tag>\
                        ""","lxml-xml")
print(root,"\n")
print(root.tag.subtag,"\n")
print(root.tag.openclosetag.attrs)
print('\n')
print(root.tag.find_all("subtag"))
print('\n')

"""
response = requests.get("http://www.cmu.edu")
root = BeautifulSoup(response.content,"lxml")
for div in root.find_all("div",class_="events"):
    for li in div.find_all("li"):
        print(li.text.strip())
print("\n")
"""
import re
text = "This course will introduce the basics of data science"
match = re.search(r"data science",text)
print(match)
print(match.start())
print("\n")

match2 = re.match(r"data scienceee",text)
print(match2)
for match in re.finditer(r"i",text):
    print(match.start())
    print(match)
for match2 in re.findall(r"i",text):
    print(match2)
print(re.findall(r"i",text))
print("\n")

regex = re.compile(r"data science")
print(regex.search(text))
print(re.search(r"[Dd]ata\s[Ss]cience",text))
print("\n")

print(re.match("\w+\s+science","data science"))
print(re.match("\w+\s+science","life science"))
print(re.match("\w+\s+science","0213123_afa science"))
print("\\")
print(r"\\")
print("\n")

match = re.search(r"(\w+)\s([Ss]cience)",text)
print(match.groups())
print(match.group(0))
print(match.group(1))
print(match.group(2))

print(re.sub(r"data science",r"data xxffdi",text))
print(re.sub(r"(\w+) ([Ss])cience",r"\1 \2chmien",text))
print(re.sub(r"(\w+) ([Ss]cience)",r"\1 \2chnieeiie","Life Science"))

print(re.match(r"abc|def","abc"))
print(re.match(r"abc|def","def"))

print(re.match(r"abc|def","abdef"))
print(re.match(r"ab(c|d)ef","abdef"))

print(re.match(r"ab(c|d)ef","abdef").groups())
print(re.match(r"ab(?:c|d)ef","abdef").groups())
