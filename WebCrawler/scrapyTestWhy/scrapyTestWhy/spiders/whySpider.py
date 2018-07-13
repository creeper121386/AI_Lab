# -*- coding: utf-8 -*-
import scrapy


class WhyspiderSpider(scrapy.Spider):
    name = 'whySpider'
    allowed_domains = ['gelbooru.com']
    start_urls = ['https://gelbooru.com/index.php?page=post&s=list&tags=landscape+sunset']

    def parse(self, response):
        for x in response.xpath(r'//*[@id=".*"]'):
            print(x)



