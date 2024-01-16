
def correct_spaces(result):	

    for i in range(len(result)):
        s = ''
        prev = ''
        for char in result[i]:
            if char == '<':
                s += ' ' + char
            elif char == "'":
                if prev == 'n':
                    s = s[:-1] + ' ' + prev + char
                else:
                    s += ' ' + char
            else:
                s += char

            prev = char

        result[i] = s

    return result


def post_process(text):

    text = text.strip()
    if len(text) > 9:
        if text[:9] != '<triplet>':
            text = '<triplet> ' + text

    return text


def decode_pred_triplets(text):
    
    triplets = []

    text = text.replace("<pad>", "").replace("</s>", "")

    text_processed = text
    text_processed = text_processed.replace("<s>", "").replace("<pad>", "").replace("</s>", "")


    for s in text_processed.split(' [SSEP] '):
        if '<aspect>' in s and '<sentiment>' in s:
            at = s.split('<sentiment>')[0].split('<aspect>')[1].strip()
            sp = s.split('<sentiment>')[1].strip()
        else:
            #print(f'Cannot decode: {s}')
            at, ot, sp = '', '', ''

        entry = {"aspect": at.strip(), "sentiment": sp.strip()}
        if entry not in triplets:
            triplets.append(entry)

    return triplets



def get_gold_triplets(dev_target_sample):

    triplets = dev_target_sample.split('|')
    triplets_list = []
    for triplet in triplets:
        d = {}
        a, s = triplet.split(';')
        d['aspect'] = a.strip()
        d['sentiment'] = sent_map[s.strip()]
        triplets_list.append(d)

    return triplets_list



def is_full_match(triplet, triplets):

    for t in triplets:
        if t['aspect'].lower() == triplet["aspect"].lower() and \
        t['sentiment'].lower() == triplet['sentiment'].lower():
            return True

    return False



def get_f1_for_trainer(predictions, target, component=None):

    # print(predictions)
    # print(target)

    n = len(target)
    assert n == len(predictions)

    preds, gold = [], []  
    for i in range(n):	
        preds.append(decode_pred_triplets(predictions[i]))
        gold.append(decode_pred_triplets(target[i]))

    pred_count = 0
    gold_count = 0
    correct_count = 0
    acc = 0

    for i in range(n):

        pred_aspects = list(set([t['aspect'].lower() for t in preds[i]]))
        #pred_opinions = list(set([t['category'].lower() for t in preds[i]]))
        gold_aspects = list(set([t['aspect'].lower() for t in gold[i]]))
        #gold_opinions = list(set([t['category'].lower() for t in gold[i]]))

        if component == 'aspect':
            pred_count += len(pred_aspects)
            gold_count += len(gold_aspects)
            correct_count += len(list(set(pred_aspects).intersection(set(gold_aspects))))

    # 		elif component == 'category':
    # 			pred_count += len(pred_opinions)
    # 			gold_count += len(gold_opinions)
    # 			correct_count += len(list(set(pred_opinions).intersection(set(gold_opinions))))

        elif component == 'sentiment':
            pair_count = 0
            triplet_count = 0
            for g in gold[i]:
                for p in preds[i]:
                    if g['aspect'] == p['aspect']: #and g['category'] == p['category']:
                        pair_count += 1
                        if g['sentiment'] == p['sentiment']:
                            triplet_count += 1
            acc += 0 if pair_count == 0 else float(triplet_count)/(pair_count)

        elif component is None:
            pred_count += len(preds[i])
            gold_count += len(gold[i])

            for gt_triplet in gold[i]:
                if is_full_match(gt_triplet, preds[i]):
                    correct_count += 1			

    if component == 'sentiment':
        acc = round(acc / n, 3)
        return acc
    else:
        p = float(correct_count) / (pred_count + 1e-8 )
        r = float(correct_count) / (gold_count + 1e-8 )
        f1 = (2 * p * r) / (p + r + 1e-8)
        return p, r, f1