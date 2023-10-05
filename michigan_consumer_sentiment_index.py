def getCSI():
    def toDate(rawDate):
        Date = ""
        rawDate = rawDate.split(" ")
        if len(rawDate[0][:-1]) == 1:
            rawDate[0] = "0" + rawDate[0]
        Date = rawDate[1] + "-" + rawDate[0][:-1]
        return Date
    import requests
    from bs4 import BeautifulSoup

    url = "https://www.mql5.com/ko/economic-calendar/united-states/michigan-consumer-sentiment"
    response = requests.get(url)

    soup = BeautifulSoup(response.text,'html.parser')

    event_history = soup.find('div', id='eventHistory')

    event_items = event_history.find_all('div', class_='event-table-history__item')

    res = {}
    for item in event_items:
        period = item.find('div', class_='event-table-history__period').text.strip()
        actual_value = float(item.find('span', class_='event-table-history__actual__value ec-tooltip-value').text.strip())
        if "예비" not in period:
            res[toDate(period)] = actual_value
    return res