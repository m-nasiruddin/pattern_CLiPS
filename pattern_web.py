#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
from pattern.web import Twitter, plaintext


twitter_en = Twitter(language='en')
for tweet_en in twitter_en.search('"more important than"', cached=False):
    print plaintext(tweet_en.text)
twitter_bn = Twitter(language='bn')
for tweet_bn in twitter_bn.search('"বিদেশে চাকরি"', cached=False):
    print plaintext(tweet_bn.text)
