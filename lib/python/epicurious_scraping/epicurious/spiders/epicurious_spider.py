# -*- coding: utf-8 -*-
import scrapy
from epicurious.items import EpicuriousItem
from math import ceil


class EpicuriousSpiderSpider(scrapy.Spider):
    name = 'epicurious_spider'
    allowed_domains = ['epicurious.com']
    start_urls = ['https://www.epicurious.com/search/?content=recipe&sort=newest']

    for i in range(1994):
        start_urls.append('https://www.epicurious.com/search/?content=recipe&sort=newest&page={}'.format(i))
        # 'https://epicurious.com/search/?meal=dinner&content=recipe']


    def parse(self, response):
        # hrefs = response.xpath("//a[contains(@class, 'view-complete-item')]//@href").extract()
        hrefs = response.xpath("//article[contains(@class, 'recipe-content-card')]//h4[contains(@class, 'hed')]//@href").extract()
        review_counts = response.xpath("//dd[contains(@class, 'reviews-count')]//text()").extract()

        # for href in hrefs:
        #     url = 'https://www.epicurious.com{}'.format(href)
        #     yield scrapy.Request(url, callback=self.parse_item)

        for href, review_count in zip(hrefs, review_counts):
            if int(review_count) >= 2:
                url = 'https://www.epicurious.com{}'.format(href)
                yield scrapy.Request(url, callback=self.parse_item)


    def parse_item(self, response):
        item = EpicuriousItem()
        
        #overall info
        item['name'] = response.xpath("//h1[contains(@itemprop, 'name')]//text()").extract_first()
        item['authors'] = response.xpath("//a[contains(@class, 'contributor')]//text()").extract()
        # item['categpry']

        #details
        item['ingredients'] = response.xpath("//li[contains(@itemprop, 'ingredients')]//text()").extract()
        item['recipe_steps'] = " ".join(response.xpath("//li[contains(@class, 'preparation-step')]//text()").extract()).strip()
        item['description'] = response.xpath("//div[contains(@itemprop, 'description')]//text()").extract_first()
        item['tags'] = response.xpath("//dl[contains(@class, 'tags')]//text()").extract()
        item['num_servings'] = response.xpath("//dd[contains(@itemprop, 'recipeYield')]//text()").extract_first()
        item['active_time'] = response.xpath("//dd[contains(@class, 'active-time')]//text()").extract_first()
        item['total_time'] = response.xpath("//dd[contains(@class, 'total-time')]//text()").extract_first()

        # nutrition
        item['calories'] = response.xpath("//span[contains(@itemprop, 'calories')]//text()").extract_first()
        item['carbs'] = response.xpath("//span[contains(@itemprop, 'carbohydrateContent')]//text()").extract_first()
        item['fat'] = response.xpath("//span[contains(@itemprop, 'fatContent')]//text()").extract_first()
        item['protein'] = response.xpath("//span[contains(@itemprop, 'proteinContent')]//text()").extract_first()
        item['saturated_fat'] = response.xpath("//span[contains(@itemprop, 'saturatedFatContent')]//text()").extract_first()
        item['sodium'] = response.xpath("//span[contains(@itemprop, 'sodiumContent')]//text()").extract_first()
        item['fiber'] = response.xpath("//span[contains(@itemprop, 'fiberContent')]//text()").extract_first()
        item['cholesterol'] = response.xpath("//span[contains(@itemprop, 'cholesterolContent')]//text()").extract_first()
        
        # ratings
        item['num_reviews'] = len(response.xpath("//div[contains(@class, 'review-text')]//p//text()").extract())
        item['make_again_pct'] = response.xpath("//div[contains(@class, 'prepare-again-rating')]//span//text()").extract_first()
        item['rating'] = response.xpath("//div[contains(@class, 'review-rating')]/span[contains(@class, 'rating')]//text()").extract_first()

        yield item