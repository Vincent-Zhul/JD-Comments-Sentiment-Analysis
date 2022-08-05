# -*- coding: utf-8 -*-

# Define here the models for your spider middleware
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/spider-middleware.html

from scrapy import signals
import requests
import json
import ssl

class JdspiderSpiderMiddleware(object):
    # Not all methods need to be defined. If a method is not defined,
    # scrapy acts as if the spider middleware does not modify the
    # passed objects.

    @classmethod
    def from_crawler(cls, crawler):
        # This method is used by Scrapy to create your spiders.
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_spider_input(response, spider):
        # Called for each response that goes through the spider
        # middleware and into the spider.

        # Should return None or raise an exception.
        return None

    def process_spider_output(response, result, spider):
        # Called with the results returned from the Spider, after
        # it has processed the response.

        # Must return an iterable of Request, dict or Item objects.
        for i in result:
            yield i

    def process_spider_exception(response, exception, spider):
        # Called when a spider or process_spider_input() method
        # (from other spider middleware) raises an exception.

        # Should return either None or an iterable of Response, dict
        # or Item objects.
        pass

    def process_start_requests(start_requests, spider):
        # Called with the start requests of the spider, and works
        # similarly to the process_spider_output() method, except
        # that it doesn’t have a response associated.

        # Must return only requests (not items).
        for r in start_requests:
            yield r

    def spider_opened(self, spider):
        spider.logger.info('Spider opened: %s' % spider.name)

class ChangeProxy(object):

    def __init__(self):
        '''
        Initialize variable
        get_url is request api
        temp_url is the verified address
        ip_list is IP
        '''
        ssl._create_default_https_context = ssl._create_unverified_context
        self.get_url = "http://api.xdaili.cn/xdaili-api//greatRecharge/getGreatIp?spiderId=e805d7cd7baa41ef87dfc5cec0ed9614&orderno=YZ2017452248ENnqkq&returnType=2&count=10"
        self.ip_list = []

        # It is used to record the number of IPS used, or the number of IPS currently being used. In this program, I get 10 IPS at one time, so the maximum default count is 9
        self.count = 0
        # It is used to record the usage times of each IP. I set this program to use it for a maximum of 4 times to replace the next IP
        self.evecount = 0

    def getIPData(self):
        '''
        This part is to obtain IP and put it into the IP pool (empty the original IP pool first
        :return:
        '''
        print("***********OBTAIN IP***********")
        temp_data = requests.get(url=self.get_url).text
        self.ip_list.clear()
        for eve_ip in json.loads(temp_data)["RESULT"]:
            self.ip_list.append({
                "ip":eve_ip["ip"],
                "port":eve_ip["port"]
            })

    def changeProxy(self,request):
        '''
        Modify proxy IP
        :param request: object
        :return:
        '''
        print("***********Modify IP***********")
        print(str(self.ip_list[self.count-1]["ip"]) + ":" + str(self.ip_list[self.count-1]["port"]))
        print("***********Modification Finish***********")
        request.meta["proxy"] = "http://" + str(self.ip_list[self.count-1]["ip"]) + ":" + str(self.ip_list[self.count-1]["port"])

    def ifUsed(self,request):
        '''
        Springboard for switching proxy IP
        :param request: object
        :return:
        '''
        try:
            self.changeProxy(request)
        except:
            if self.count == 0 or self.count == 9:
                self.getIPData()
                self.count = 1
                self.evecount = 0
            self.count = self.count + 1
            self.ifUsed(request)


    def process_request(self, request, spider):

        if self.count == 0 or self.count==9:
            self.getIPData()
            self.count = 1

        if self.evecount == 40:
            self.count = self.count + 1
            self.evecount = 0
            self.ifUsed(request)
        else:
            self.evecount = self.evecount + 1


