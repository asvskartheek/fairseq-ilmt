from . import DATASET_REGISTRY
from . import dataset_register, data_abspath
from . import Corpus, sanity_check
from itertools import permutations, combinations


@dataset_register('iitb-hi-en', ['train', 'dev', 'test'])
def IITB_meta(split):
    corpora = []
    for lang in ['en', 'hi']:
        sub_path = 'filtered-iitb/{}.{}'.format(split, lang)
        corpus = Corpus('iitb-hi-en', data_abspath(sub_path), lang)
        corpora.append(corpus)
    return corpora

@dataset_register('national-newscrawl', ['train', 'dev', 'test'])
def NationalNewscrawl_meta(split):
    if split in ['dev', 'test']:
        return []
    corpora = []
    for lang in ['en', 'hi']:
        sub_path = 'national-newscrawl/national.{}'.format(lang)
        #corpus = Corpus('iitb-hi-en', data_abspath(sub_path), lang)
        corpus = Corpus('national-newscrawl', data_abspath(sub_path), lang)
        corpora.append(corpus)
    return corpora

@dataset_register('wat-ilmpc', ['train', 'dev', 'test'])
def WAT_meta(split):
    corpora = []
    langs = ['bn', 'hi', 'ml', 'ta', 'te', 'ur']
    for lang in langs:
        for src in [lang, 'en']:
            sub_path = 'indic_languages_corpus/bilingual/{}-en/{}.{}'.format(
                    lang, split, src
            )
            corpus_name = 'wat-ilmpc-{}-{}'.format(lang, 'en')
            corpus = Corpus(corpus_name, data_abspath(sub_path), src)
            corpora.append(corpus)
    return corpora

@dataset_register('pib-test', ['test'])
def PIBTEST_meta(split):
    corpora = []
    langs = ['hi', 'ta', 'te', 'ml', 'ur', 'bn', 'gu', 'mr', 'pa', 'or']
    for lang in langs:
        fst, snd = sorted([lang, 'en'])
        dirname = '{}-{}'.format(fst, snd)
        for src in [lang, 'en']:
                sub_path = 'pib-test/{}/{}.{}'.format(
                        dirname, split, src
                )
                corpus_name = 'pib-test-{}-{}'.format(lang, 'en')
                corpus = Corpus(corpus_name, data_abspath(sub_path), src)
                corpora.append(corpus)
    return corpora

@dataset_register('pib', ['train', 'dev', 'test'])
def PIB_meta(split):
    if split in ['dev', 'test']:
        return []

    corpora = []
    langs = ['hi', 'ta', 'te', 'ml', 'ur', 'bn', 'gu', 'mr', 'pa', 'or', 'en']
    langs = sorted(langs)
    perm = combinations(langs, 2)
    for src, tgt in list(perm):
        for lang in [src, tgt]:
                sub_path = 'pib/{}-{}/{}.{}'.format(
                        src, tgt, 'train', lang
                )
                corpus_name = 'pib-{}-{}'.format(src, tgt)
                corpus = Corpus(corpus_name, data_abspath(sub_path), lang)
                corpora.append(corpus)
    return corpora

@dataset_register('pib-v1', ['train', 'dev', 'test'])
def PIBV2_meta(split):
    if split in ['dev', 'test']:
        return []

    corpora = []
    langs = ['hi', 'ta', 'te', 'ml', 'ur', 'bn', 'gu', 'mr', 'pa', 'or', 'en']
    langs = sorted(langs)
    perm = combinations(langs, 2)
    for src, tgt in list(perm):
        for lang in [src, tgt]:
                sub_path = 'pib-v1/{}-{}/{}.{}'.format(
                        src, tgt, 'train', lang
                )
                corpus_name = 'pib-v1-{}-{}'.format(src, tgt)
                corpus = Corpus(corpus_name, data_abspath(sub_path), lang)
                corpora.append(corpus)
    return corpora

@dataset_register('pib-v0.2', ['train', 'dev', 'test'])
def PIBV02_meta(split):
    if split in ['dev', 'test']:
        return []

    corpora = []
    langs = ['hi', 'ta', 'te', 'ml', 'ur', 'bn', 'gu', 'mr', 'pa', 'or', 'en']
    langs = sorted(langs)
    perm = combinations(langs, 2)
    for src, tgt in list(perm):
        for lang in [src, tgt]:
                sub_path = 'pib-v0.2/{}-{}/{}.{}'.format(
                        src, tgt, 'train', lang
                )
                corpus_name = 'pib-v0.2-{}-{}'.format(src, tgt)
                corpus = Corpus(corpus_name, data_abspath(sub_path), lang)
                corpora.append(corpus)
    return corpora

@dataset_register('mkb-v0', ['train', 'dev', 'test'])
def MKB_meta(split):
    if split in ['train', 'dev']:
        return []

    corpora = []
    langs = ['ml', 'ur', 'te', 'hi', 'ta', 'bn', 'gu', 'or', 'mr', 'en']
    langs = sorted(langs)
    perm = combinations(langs, 2)
    for src, tgt in list(perm):
        for lang in [src, tgt]:
                sub_path = 'mkb/{}-{}/{}.{}'.format(
                        src, tgt, 'mkb', lang
                )
                corpus_name = 'mkb-v0-{}-{}'.format(src, tgt)
                corpus = Corpus(corpus_name, data_abspath(sub_path), lang)
                corpora.append(corpus)
    return corpora


@dataset_register('wat-mkb-dev', ['train', 'dev', 'test'])
def WAT_MKB_DEV(split):
    if split in ['train']:
        return []

    corpora = []
    langs = ['bn', 'gu', 'hi', 'ml', 'mr', 'ta', 'te']
    for lang in langs:
        for src in [lang, 'en']:
            sub_path = 'wat-mkb-dev/{}.{}-en.{}'.format(
                    split, lang, src
            )
            corpus_name = 'wat-mkb-dev-{}-{}'.format(lang, 'en')
            corpus = Corpus(corpus_name, data_abspath(sub_path), src)
            corpora.append(corpus)
    return corpora

@dataset_register('wat-mkb-test', ['train', 'dev', 'test'])
def WAT_MKB_TEST(split):
    if split in ['train', 'dev']:
        return []

    corpora = []
    langs = ['bn', 'gu', 'hi', 'ml', 'mr', 'ta', 'te']
    for lang in langs:
        for src in [lang, 'en']:
            sub_path = 'wat-mkb-test/{}.{}-en.{}'.format(
                    split, lang, src
            )
            corpus_name = 'wat-mkb-test-{}-{}'.format(lang, 'en')
            corpus = Corpus(corpus_name, data_abspath(sub_path), src)
            corpora.append(corpus)
    return corpora

@dataset_register('ufal-en-tam', ['train', 'dev', 'test'])
def UFALEnTam_meta(split):
    corpora = []
    for lang in ['en', 'ta']:
        sub_path = 'ufal-en-tam/{}.{}.{}'.format('corpus.bcn', split, lang)
        corpus = Corpus('ufal-en-tam', data_abspath(sub_path), lang)
        corpora.append(corpus)
    return corpora


@dataset_register('ilci', ['train', 'dev', 'test'])
def ILCI_meta(split):
    #if split in ['dev', 'test']:
    #    return []
    corpora = []
    langs = [
        'en', 'te', 'hi', 'ml', 
        'ta', 'ud', 'bg', 'mr',
        'gj', 'pj', 'kn'
    ]

    from .utils import canonicalize

    for lang in langs:
        sub_path = 'ilci/{}.{}'.format(split, lang)
        _lang = canonicalize(lang)
        corpus = Corpus('ilci', data_abspath(sub_path), _lang)
        corpora.append(corpus)
    return corpora

@dataset_register('bible-en-te', ['train', 'dev', 'test'])
def BIBLEEnTe_meta(split):
    corpora = []
    for lang in ['en', 'te']:
        sub_path = 'bible-en-te/{}.{}.{}'.format('bible', split, lang)
        corpus = Corpus('bible-en-te', data_abspath(sub_path), lang)
        corpora.append(corpus)
    return corpora


@dataset_register('eenadu-en-te', ['train'])
def EenaduBacktrans_meta(split):
    if split in ['dev', 'test']:
        return []

    corpora = []
    for lang in ['en','te']:
        sub_path = 'eenadu-en-te/train.{}'.format(lang)
        corpus = Corpus('eenadu-en-te', data_abspath(sub_path), lang)
        corpora.append(corpus)
    return corpora

@dataset_register('odiencorp', ['train', 'dev', 'test'])
def OdiEnCorp_meta(split):
    corpora = []
    for lang in ['en', 'or']:
        sub_path = 'odiencorp/{}.{}'.format(split, lang)
        corpus = Corpus('odiencorp', data_abspath(sub_path), lang)
        corpora.append(corpus)
    return corpora


@dataset_register('newstest2019guen', ['train', 'dev', 'test'])
def NewsTest2019guen_meta(split):
    if split in ['train', 'dev']:
        return []

    corpora = []
    for lang in ['en','gu']:
        sub_path = 'newstest2019guen/{}.{}'.format(split, lang)
        corpus = Corpus('newstest2019guen', data_abspath(sub_path), lang)
        corpora.append(corpus)
    return corpora

@dataset_register('newstest2019engu', ['train', 'dev', 'test'])
def NewsTest2019engu_meta(split):
    if split in ['train', 'dev']:
        return []

    corpora = []
    for lang in ['en','gu']:
        sub_path = 'newstest2019engu/{}.{}'.format(split, lang)
        corpus = Corpus('newstest2019engu', data_abspath(sub_path), lang)
        corpora.append(corpus)
    return corpora

@dataset_register('newstest2020taen', ['train', 'dev', 'test'])
def NewsTest2020taen_meta(split):
    if split in ['train', 'dev']:
        return []

    corpora = []
    for lang in ['en','ta']:
        sub_path = 'newstest2020taen/{}.{}'.format(split, lang)
        corpus = Corpus('newstest2020taen', data_abspath(sub_path), lang)
        corpora.append(corpus)
    return corpora

@dataset_register('newstest2020enta', ['train', 'dev', 'test'])
def NewsTest2020enta_meta(split):
    if split in ['train', 'dev']:
        return []

    corpora = []
    for lang in ['en','ta']:
        sub_path = 'newstest2020enta/{}.{}'.format(split, lang)
        corpus = Corpus('newstest2020enta', data_abspath(sub_path), lang)
        corpora.append(corpus)
    return corpora

@dataset_register('newsdev2019', ['train', 'dev', 'test'])
def NewsDev2019_meta(split):
    if split in ['train']:
        return []

    corpora = []
    langs = ['en', 'gu']
    for lang in langs:
        sub_path = 'newsdev2019/{}.{}'.format(split, lang)
        corpus = Corpus('newsdev2019', data_abspath(sub_path), lang)
        corpora.append(corpus)
    return corpora

@dataset_register('newsdev2020', ['train', 'dev', 'test'])
def NewsDev2020_meta(split):
    if split in ['train']:
        return []

    corpora = []
    langs = ['en','ta']
    for lang in langs:
        sub_path = 'newsdev2020/{}.{}'.format(split, lang)
        corpus = Corpus('newsdev2020', data_abspath(sub_path), lang)
        corpora.append(corpus)
    return corpora

@dataset_register('complete-en-ml', ['train', 'dev', 'test'])
def CompleteEnMl_meta(split):
    corpora = []
    for lang in ['en', 'ml']:
        sub_path = 'complete-en-ml/{}.ml-en.{}'.format(split, lang)
        corpus = Corpus('complete-en-ml', data_abspath(sub_path), lang)
        corpora.append(corpus)
    return corpora


if __name__ == '__main__':
    def merge(*_as):
        _ase = []
        for a in _as:
            _ase.extend(a)
        return _ase

    ls = []
    for key in DATASET_REGISTRY:
        splits, f = DATASET_REGISTRY[key]
        for split in splits:
            ls.append(f(split))

    _all = merge(*ls)
    sanity_check(_all)

