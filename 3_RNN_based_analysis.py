# выполнил Субботин Дмитрий, группа 932001

text = """У лукоморья дуб зелёный;
      Златая цепь на дубе том:
      И днём и ночью кот учёный
      Всё ходит по цепи кругом;
      Идёт направо — песнь заводит,
      Налево — сказку говорит.
      Там чудеса: там леший бродит,
      Русалка на ветвях сидит;
      Там на неведомых дорожках
      Следы невиданных зверей;
      Избушка там на курьих ножках
      Стоит без окон, без дверей;
      Там лес и дол видений полны;
      Там о заре прихлынут волны
      На брег песчаный и пустой,
      И тридцать витязей прекрасных
      Чредой из вод выходят ясных,
      И с ними дядька их морской;
      Там королевич мимоходом
      Пленяет грозного царя;
      Там в облаках перед народом
      Через леса, через моря
      Колдун несёт богатыря;
      В темнице там царевна тужит,
      А бурый волк ей верно служит;
      Там ступа с Бабою Ягой
      Идёт, бредёт сама собой,
      Там царь Кащей над златом чахнет;
      Там русский дух… там Русью пахнет!
      И там я был, и мёд я пил;
      У моря видел дуб зелёный;
      Под ним сидел, и кот учёный
      Свои мне сказки говорил."""

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

from rnnmorph.predictor import RNNMorphPredictor

tokenz = word_tokenize(text)

m = RNNMorphPredictor(language='ru')

data = m.predict(tokenz)

# Приведение строк tag, которые содержат морфологический разбор слова, к типу данных словарь
tags = []

for analysis in data:
  tag_string = analysis.tag.split('|')

  tag = {}
  for s in tag_string:
    if (len(s) > 1):
      tmp = s.split('=')
      tag[tmp[0]] = tmp[1]
  tags.append(tag)

  result = set()

# Поиск слов
req_parts = ['NOUN', 'ADJ']

for word1, word2, tag1, tag2 in zip(data, data[1:], tags, tags[1:]):
  # имена существительные или имена прилагательные на первом или втором месте
  if (word1.pos in req_parts) or (word2.pos in req_parts):
    # слова совпадают по роду, числу и падежу
    if ('Gender' in tag1) and ('Gender' in tag2) and (tag1['Gender'] == tag2['Gender']):
      if ('Number' in tag1) and ('Number' in tag2) and (tag1['Number'] == tag2['Number']):
        if ('Case' in tag1) and ('Case' in tag2) and (tag1['Case'] == tag2['Case']):
          result.add(word1.normal_form + ' ' + word2.normal_form)

print(result)

# ВЫВОД
# {'баба яга',
#  'бурый волк',
#  'грозный царь',
#  'дуб зелёный',
#  'дуб тот',
#  'златой цепь',
#  'кот учёный',
#  'русский дух…'}