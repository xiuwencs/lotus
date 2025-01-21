import random
import re
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from collections import Counter
from PIL import Image, ImageDraw



with open("http.txt", "r") as f:
    messages = f.readlines()

vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(messages)

dbscan = DBSCAN(eps=0.2, min_samples=1)#将消息聚类
dbscan.fit(x)

clusters = {}
for i, label in enumerate(dbscan.labels_):
    if label not in clusters:
        clusters[label] = []
    clusters[label].append((messages[i].strip()))

with open("http_cluster.txt", "w") as f:
    for label in sorted(clusters.keys()):
        f.write("cluster {}:{}messages\n".format(label, len(clusters[label])))
        for message in clusters[label]:
            f.write(message + "\n")
        f.write("\n")


def group_fields(pos_list):
    """
           根据词性组合字段
           :param pos_list：词性列表
           :return: 以词性形式展示的字段划分结果
           """
    seg_list = []
    i = 0
    # print("len:",len(pos_list))
    while i < len(pos_list) - 1:
        if pos_list[0] == 'CD':  # smtp command
            seg_list.append([pos_list[0]])
            seg_list.append(pos_list[0:])
            break
        pos = pos_list[i]
        if pos == 'NN' and i < len(pos_list) - 1 and pos_list[i + 1] == pos:
            seg_list.append([pos, pos_list[i + 1]])
            i += 1
        if pos == 'RB' and i < len(pos_list) - 1 and pos_list[i + 1] == 'VBN':
            seg_list.append([pos, pos_list[i + 1]])
            i += 1
        elif pos == ":":
            start = i - 1
            while i < len(pos_list) - 1 and pos_list[i + 1] != ":":
                i += 1
            if i == len(pos_list) - 1:
                end = i + 1
            else:
                end = i
            seg_list.append(pos_list[start:end])
            # print("rule2:", seg_list)
        elif pos == ",":
            start = i - 1
            while i < len(pos_list) - 1 and pos_list[i + 1] != ",":
                i += 1
            end = i + 2
            seg_list.append(pos_list[start:end])
        # smtp ftp pass
        elif pos_list[i + 1] == ":" or pos_list[i + 1] == ",":
            i += 0
        #
        else:
            seg_list.append([pos])
        i += 1
    return seg_list


def convert_to_token(seg_list, token_list):
    """
           将词性表示的字段转化为字符表示的字段
           :param seg_list:词性表示的字段
           :param token_list:字符列表
           :return: 以字符形式展示的字段划分结果
           """
    seg_token = []
    for item in seg_list:
        if isinstance(item, list):
            sub_token_list = []
            for pos in item:
                if token_list:
                    sub_token_list.append(token_list.pop(0))
            seg_token.append(sub_token_list)
        else:
            seg_token.append(token_list.pop(0))
    return seg_token


def convert_to_hex(token_list):
    """
               将字符表示的字段转化为十六进制表示的字段
               :param token_list:字符列表
               :return: 以十六进制形式展示的字段划分结果
               """
    hex_token_list = []
    for token in token_list:
        if isinstance(token, list):
            hex_token_list.append(convert_to_hex(token))
        else:
            hex_token = ''
            for char in token:
                hex_char = hex(ord(char))[2:]
                hex_token += hex_char
            hex_token_list.append(hex_token)
    if not hex_token_list:
        return []
    result_seg_list = [hex_token_list[0]]
    for sublist in hex_token_list[1:]:
        is_duplicate = False
        for existing_sublist in result_seg_list:
            if all(item in existing_sublist for item in sublist):
                is_duplicate = True
                break
        if not is_duplicate:
            result_seg_list.append(sublist)
    return result_seg_list


def add_missing_chars(original_message, str_message):
    """
               添加在分词过程中缺失的十六进制字段，比如空格和回车
               :param original_message：原始消息
               :param str_message:字段划分后的消息
               :return: 修正后的字段划分消息
               """
    result = ""
    i = 0
    j = 0
    while i < len(original_message) and j < len(str_message):
        if original_message[i] == str_message[j]:
            result += str_message[j]
            i += 1
            j += 1
        else:
            if str_message[j] == ',':
                result += ','
                j += 1
            else:
                result += original_message[i]
                i += 1
    result += original_message[i:]
    return result


def extract_chars(text):
    """
               提取带有chunk的消息的特殊字段“0d0a0d0a”，它是分割消息头和chunk的标志
               :param text：消息
               :return: 特殊字段及其偏移量
               """
    start_index = text.find("0d0a0d0a")
    end_index = text.find("0d0a0d0a", start_index + 1)
    if start_index != -1 and end_index != -1:
        chars = text[start_index:end_index + 8]
        return chars, start_index
    else:
        return None


def extract_most_frequent_string(text):
    """
                   提取频繁字符
                   :param text：消息
                   :return: 频繁字符
                   """
    global frequent_string
    substrings = re.findall(r'(?=(.{1,4}))', text)
    counter = Counter(substrings)
    most_common = counter.most_common(5)
    if most_common[0][0] == '0' * len(most_common[0][0]):
        frequent_string = most_common[1][0]
    else:
        frequent_string = most_common[0][0]
    return frequent_string



def add_frequent_strings(text):
    """
                       根据频繁字符再划分字段
                       :param text：消息
                       :return: 划分后的字段
                       """
    between_string, flag = extract_chars(text)
    if between_string:
        frequent_string = extract_most_frequent_string(between_string)
        final = between_string.replace(frequent_string, "," + frequent_string + ",")
        final = final.replace(",,", ",")
        text = text[:flag] + final
        return text


def check_flag(str_1, flag_1):
    """
                       判断频繁字符是否是之前经过分词和词性标注划分的字段边界的一部分
                       :param str_1：经过分词和词性标注划分的字段边界
                       :param flag_1:频繁字符
                       :return: True or False
                       """
    substrings = str_1.split(',')
    flag_1 = str(flag_1)
    if flag_1 in substrings:
        return True
    return False


def add_comma(text):
    """
                        对于字段划分结果中空格和回车的修正
                        :param text：消息
                        :return: 分词结果
                        """
    result1 = ""
    i = 0
    while i < len(text):
        if text[i:i + 2] == "20" or text[i:i + 4] == "0d0a":
            if i > 0 and text[i] == ",":
                result1 += ","
            result1 += text[i:i + 2] if text[i:i + 2] == "20" else text[i:i + 4]
            if i < len(text) - 2 and text[i + 2] != ',' and text[i - 1] == ",":
                result1 += ","
            i += 2 if text[i:i + 2] == "20" else 4
        else:
            result1 += text[i]
            i += 1
    result1 = result1.replace(",,", ",")

    return result1


with open("http.txt", "w") as f:
    for label in sorted(clusters.keys()):
        random_message = random.choice(clusters[label])
        #将预处理后的消息以字节隔开
        formatted_message = ' '.join(random_message[i:i + 2] for i in range(0, len(random_message), 2))
        #将十六进制转化为相应的字符形式
        ascii_message = ''.join(chr(int(hex_bytes, 16)) for hex_bytes in formatted_message.split())
        #分词
        tokens = word_tokenize(ascii_message)
        f.write("cluster{}:\n{}\n".format(label, tokens))
        # 词性标注
        pos_tags = pos_tag(tokens)
        pos_list = []
        token_list = []
        for token, pos in pos_tags:
            pos_list.append(pos)
            token_list.append(token)

        seg_list = group_fields(pos_list)
        seg_token = convert_to_token(seg_list, token_list)

        #对“；”字符的修正：与前一个字段合并
        merged_list = []
        i = 0
        while i < len(seg_token):
            if i > 0 and len(seg_token[i]) > 1 and seg_token[i][1] == ";":
                merged_list[-1].extend(seg_token[i])
            else:
                merged_list.append(seg_token[i])
            i += 1

        #对非终结符号的修正：与后一个字段合并
        new_merged_list = []
        j = 0
        non_terminal_symbols = {'(', '[', '<', '{', ':'}
        print(merged_list)
        while j < len(merged_list):
            if merged_list[j] and j < len(merged_list) - 1 and merged_list[j][-1] in non_terminal_symbols:
                new_sublist = merged_list[j] + merged_list[j + 1]
                new_merged_list.append(new_sublist)
                j += 2
            else:
                new_merged_list.append(merged_list[j])
                j += 1
        merged_list = new_merged_list


        seg_chunked_list = []
        flag_gzip = 0
        #判断是否含有chunk
        for sublist in new_merged_list:
            if 'gzip' in sublist:
                flag_gzip = 1
                # print("found gzip!")
                index = sublist.index('gzip')
                seg_chunked_list.append(sublist[:index + 1])
                seg_chunked_list.append(sublist[index + 1:])
            else:
                seg_chunked_list.append(sublist)

        seg_message = convert_to_hex(seg_token)
        new_seg_message = []
        k = 0
        non_terminal_symbols_hex = {'28', '3c', '3a'}
        while k < len(seg_message):
            if k < len(seg_message) - 1 and seg_message[k][-1] in non_terminal_symbols_hex:
                new_sub = seg_message[k] + seg_message[k + 1]
                new_seg_message.append(new_sub)
                k += 2
            else:
                new_seg_message.append(seg_message[k])
                k += 1
        seg_str = ",".join(
            ["".join(field).replace("[", "").replace("]", "").replace(" ", "") for field in new_seg_message])
        seg_str = seg_str.replace("[]", ",")


        new_str = add_missing_chars(random_message, seg_str)
        if flag_gzip == 1:
            gzip_str = add_frequent_strings(new_str)
            result = add_comma(gzip_str)
        else:
            result = add_comma(new_str)
        flag = extract_most_frequent_string(random_message)
        if check_flag(result, flag):
            result = result.replace(flag, ',' + flag + ',')
            result = result.replace(',,', ',')
        if result.endswith('0d0a'):
            result = result[:-4] + ',0d0a,'
        print("result:", result + '\n')
        # rtsp
        # if not flag_gzip:
        # stop_index = result.find(flag)
        # if stop_index != -1:
        # result = result[:stop_index + 10]
        # if result[-1] == ",":
        # result = result[:-1]
        # print("result:", result + '\n')

        # print("original_message_frequent:", flag)
