import os
import jieba
import pickle

dataset = 'microblogPCU'
stop_word_list = [' ', '，', '：', '。', '.', '/', ':', '》', '《', '！', '-']

if __name__ == "__main__":
    if dataset == 'microblogPCU':
        word_idx_map = dict()
        all_data = []
        user_id_map = dict()
        with open(f'dataset/{dataset}/weibo_user.csv', 'rb') as f_user:
            next(f_user)
            valid_cnt, err_cnt = 0, 0
            for row in f_user.readlines():
                cur_data = dict()
                try:
                    user_id, _, _, gender, uclass, message, post_num, follower_num, followee_num, is_spammer = \
                        row.decode('gb18030').strip().split(',')
                    gender, uclass = 1 if gender=='male' else 0, int(uclass)
                    message = ' '.join([word for word in message.strip().split(' ') if word!=''])
                    message_bag = set([word for word in list(jieba.cut(message)) if word not in stop_word_list])
                    post_num, follower_num, followee_num = \
                        int(post_num), int(follower_num), int(followee_num)

                    if user_id not in user_id_map:
                        user_id_map[user_id] = len(user_id_map)
                    cur_data['user_id'] = user_id_map[user_id]
                    # cur_data['feature'] = [gender, uclass, post_num, follower_num, followee_num, message_bag]
                    cur_data['feature'] = [message_bag]
                    cur_data['label'] = is_spammer
                    for word in message_bag:
                        if word not in word_idx_map:
                            word_idx_map[word] = len(word_idx_map)
                    # print(user_id, gender, uclass, message, follower_num, followee_num, is_spammer)
                    # print(cur_data)
                    all_data.append(cur_data)
                    valid_cnt += 1
                except:
                    err_cnt += 1
        print(valid_cnt, err_cnt)
        word_idx_map_len = len(word_idx_map)

        for data in all_data:
            words = data['feature'][-1]
            del data['feature'][-1]
            cur_feat = [0]*word_idx_map_len
            for word in words:
                cur_feat[word_idx_map[word]] = 1
            data['feature'].extend(cur_feat)

        with open(f'dataset/{dataset}/new_user.pkl', 'wb') as f:
            pickle.dump(all_data, f)

        with open(f'dataset/{dataset}/follower_followee.csv', 'rb') as f_user, \
                open(f'dataset/{dataset}/new_adj.csv', 'w') as f_adj:
            next(f_user)
            valid_cnt, err_cnt = 0, 0
            for row in f_user.readlines():
                _, _, follower_id, _, followee_id = row.decode('gb18030').strip().split(',')[:5]
                if follower_id not in user_id_map or followee_id not in user_id_map:
                    err_cnt += 1
                    continue
                f_adj.write(f'{user_id_map[followee_id]},{user_id_map[follower_id]}\n')
                valid_cnt += 1
        print(valid_cnt, err_cnt)
