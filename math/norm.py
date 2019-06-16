# -*- coding: utf-8 -*-

import numpy as np

from scipy.stats import norm

def sigma_from_miu_cdf(miu, x, cdf_x):
    return (x-miu)/norm.ppf(cdf_x)

def truncated_norm_expectation(a, b, miu=0, sigma=1):
    _a=(a-miu)/sigma
    _b=(b-miu)/sigma
    pdf_a, pdf_b=norm.pdf([_a, _b])
    cdf_a, cdf_b=norm.cdf([_a, _b])
    return miu+sigma*(pdf_a-pdf_b)/(cdf_b-cdf_a)

def norm_regional_expectation(a, b, miu=0, sigma=1):
    _a=(a-miu)/sigma
    _b=(b-miu)/sigma
    pdf_a, pdf_b=norm.pdf([_a, _b])
    cdf_a, cdf_b=norm.cdf([_a, _b])
    return miu*(cdf_b-cdf_a)+sigma*(pdf_a-pdf_b)
