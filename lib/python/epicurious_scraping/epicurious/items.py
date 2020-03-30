# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class EpicuriousItem(scrapy.Item):
    # define the fields for your item here like:
    name = scrapy.Field()
    authors = scrapy.Field()
    ingredients = scrapy.Field()
    recipe_steps = scrapy.Field()
    description = scrapy.Field()
    tags = scrapy.Field()
    num_servings = scrapy.Field()
    active_time = scrapy.Field()
    total_time = scrapy.Field()

    calories = scrapy.Field()
    carbs = scrapy.Field()
    fat = scrapy.Field()
    protein = scrapy.Field()
    saturated_fat = scrapy.Field()
    sodium = scrapy.Field()
    fiber = scrapy.Field()
    cholesterol = scrapy.Field()

    num_reviews = scrapy.Field()
    make_again_pct = scrapy.Field()
    rating = scrapy.Field()
