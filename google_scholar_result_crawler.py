from bs4 import BeautifulSoup
import requests, time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


def requests_retry_session(
    retries=5,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504),
    session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

_PROXIES = {
    'https': 'https://127.0.0.1:10087',
    'http': 'http://127.0.0.1:10087'
}

def get_top_K10_gs_results(query: str, top_10k):
    query = query.replace(" ", "+")
    print(query)
    titles = []
    links = ("https://scholar.google.com/scholar?start={}&q={}&hl=en&as_sdt=1,5&as_vis=1".format(10*k, query) for k in range(top_10k))
    for i, l in enumerate(links):
        rsp = requests_retry_session().get(l)
        if rsp.status_code == 429:
            print("Too many requests. Sleep for 120s.")
            time.sleep(120)
            rsp = requests_retry_session().get(l)
        if rsp.status_code == 200:
            bsObj = BeautifulSoup(rsp.text, features="html.parser")
            tags = bsObj.find_all('h3', attrs={'class': 'gs_rt'})
            for tag in tags:
                ta = tag.find_all('a')
                res = ta[0].get_text().strip('.')
                titles.append(res+"\n")
        else:
            print("Error {} with request for: {}".format(rsp.status_code, l))
        print("top {} retrieved.".format(10*(i+1)))
    with open("query_results/{}_top_{}.txt".format(query, top_10k * 10), "w", encoding="utf-8") as rw:
        rw.writelines(titles)

if __name__ == '__main__':
    with open("queries.txt", "r", encoding="utf-8") as qsfr:
        for l in qsfr.readlines():
            l = l.strip()
            get_top_K10_gs_results(l, 10)