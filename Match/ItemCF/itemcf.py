"""
    ItemCF的实现

    Reference:
    [1] 推荐系统实战 项亮
    [2] https://github.com/wangzhegeek/item-userCF
    [3] https://github.com/ZiyaoGeng/SimpleCF
    [4] https://github.com/xingzhexiaozhu/MovieRecommendation
    [5] https://github.com/xiaogp/recsys_spark
    [6] 基于物品的协同过滤i2i--算法、trick及分布式实现 https://zhuanlan.zhihu.com/p/350447698

"""

import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from metrics import MatchMetrics
import math
import threading
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from tqdm import tqdm


class ItemCF:
    def __init__(self, train_data, user_col_name, item_col_name, label_col_name, max_workers=None, norm=False):
        """

        Args:
            train_data (pd.DataFrame): 训练数据, 包含user, item, label三列, 且user, item已经由LabelEncoder编码完毕
            user_col_name (str): 用户列名
            item_col_name (str): 物料列名
            label_col_name (str): label列名
            max_workers (int): 线程数
            norm (bool): 是否对item相似矩阵做norm处理
        """

        self.data = train_data
        self.user_col_name = user_col_name
        self.item_col_name = item_col_name
        self.label_col_name = label_col_name
        self.max_workers = max_workers if max_workers else os.cpu_count() * 5
        self.norm = norm
        # user数量
        self.n_user = train_data[user_col_name].nunique()
        # item数量
        self.n_item = train_data[item_col_name].nunique()

        # 多线程锁
        self.lock = threading.Lock()

        # item相似度矩阵
        print("正在生成item相似度矩阵...")
        self._get_item_similarity()
        print("生成完毕")

    def _get_user_item_index(self):
        """
            生成user-item的倒排索引, 以pd.DataFrame为容器
        Returns:
            user-item的倒排索引
        """

        user_item = self.data.groupby(self.user_col_name).apply(lambda x: list(x[self.item_col_name]))

        return user_item

    def _get_item_similarity(self):
        """
            生成item相似性字典,
        Returns:
            item相似性矩阵
        """

        # 每个item被交互的次数
        item_user_n = self.data.groupby(self.item_col_name).agg(
            users_n=pd.NamedAgg(column=self.item_col_name, aggfunc='count'))

        # 构建user-item的倒排索引
        user_item = self._get_user_item_index()

        # item相似度
        self.item_sim_dict = dict()

        def _calc_sim_partial(user_item):
            """
                得到item cos相似度的分子部分
            Args:
                user_item (pd.Series): user-item的倒排索引

            Returns:

            """

            for index, user_item_list in tqdm(user_item.iteritems(), total=user_item.count()):
                # item倒排索引列表长度
                user_item_list_length = len(user_item_list)
                # i,j为user_item_list的item索引下标
                for i in range(len(user_item_list) - 1):
                    item_i = user_item_list[i]
                    self.item_sim_dict.setdefault(item_i, {})
                    for j in range(i + 1, len(user_item_list)):
                        item_j = user_item_list[j]
                        self.item_sim_dict.setdefault(item_j, {})

                        self.item_sim_dict[item_i].setdefault(item_j, 0.0)
                        self.item_sim_dict[item_j].setdefault(item_i, 0.0)

                        # 热门惩罚
                        i_j_sim = 1.0 / math.log(1.0 + user_item_list_length)

                        # 多个线程写同一个对象, 需要全局锁
                        self.lock.acquire()
                        # 对称位置都要计算并更新
                        self.item_sim_dict[item_i][item_j] += i_j_sim
                        self.item_sim_dict[item_j][item_i] += i_j_sim
                        # 释放锁
                        self.lock.release()

        # 实际上由于锁的存在，多线程的效率不如单线程
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            # 每个线程处理的数据量
            sample_per_thead = self.n_user // self.max_workers + 1
            # 数据范围区间
            interval = []
            left_point = 0
            for i in range(self.max_workers):
                interval.append(user_item[left_point:left_point+sample_per_thead])
                left_point += sample_per_thead
                pool.map(_calc_sim_partial, interval)

        # 除以sqrt(N(i)*N(j)), 得到最终的cos相似度
        for item_i, related_items in self.item_sim_dict.items():
            for item_j, i_j_sim in related_items.items():
                self.item_sim_dict[item_i][item_j] /= math.sqrt(
                             item_user_n.at[item_i, "users_n"] * item_user_n.at[item_j, "users_n"])

        # item相似度归一化
        if self.norm:
            for item, related_items in self.item_sim_dict.items():
                # 得到item相似度的最大值
                _, max_w = sorted(related_items.items(), key=lambda x: x[1], reverse=True)[0]
                for neibor_item in related_items.keys():
                    self.item_sim_dict[item][neibor_item] /= max_w

    def recommend(self, user, k, n):
        """
            为一个user召回item
        Args:
            user (str): trigger user
            k (int): 每个trigger item召回item的数量
            n (int): 召回item总数量

        Returns:
            召回item列表, [(item, score), ()...]
        """

        # 召回item
        item_dict = dict()
        # user的历史交互item
        user_trigger_items = list(self.data[self.data[self.user_col_name] == user][self.item_col_name])

        # 遍历user的历史交互item
        for trigger_item in user_trigger_items:
            # 遍历每一个trigger_item的最相似的k个item作为possible_item
            for possible_item, wij in sorted(self.item_sim_dict[trigger_item].items(), key=lambda x: x[1], reverse=True)[:k]:
                # 判断possible_item是否在user的历史交互item中
                if possible_item not in user_trigger_items:
                    item_dict.setdefault(possible_item, 0.0)
                    # 计算并更新possible_item分数, 1为权重, 也可自定义权重
                    item_dict[possible_item] += 1 * wij

        # 排序并取前n项
        item_dict = sorted(item_dict.items(), key=lambda x: x[1], reverse=True)[:n]  # [(item, score), ()...]

        return item_dict

    def evaluate(self, test_data, k, n):
        """
            评估在test_data上的召回指标

        Args:
            test_data: 测试数据
            k (int): 每个trigger item召回item的数量
            n (int): 召回item总数量

        Returns:
            整个测试数据集的recall, precision
        """

        recall, precision = 0, 0

        test_user_labels = test_data.groupby(self.user_col_name).apply(lambda x: list(x[self.item_col_name]))

        # 遍历测试集中每一个user
        print("开始评估测试集")
        for user, item_labels_list in tqdm(test_user_labels.iteritems(), total=test_user_labels.count()):
            # 召回items
            pred_items_list = [item for item, score in self.recommend(user, k, n)]
            if pred_items_list:     # 召回结果非空
                user_recall, user_precision = MatchMetrics.get_recall_and_precision(labels=item_labels_list, preds=pred_items_list)
                recall += user_recall
                precision += user_precision

        # 测试集指标为每个user指标的算数平均, 把召回为空的用户的准召默认为0
        recall /= len(test_user_labels)
        precision /= len(test_user_labels)

        return recall, precision

if __name__ == "__main__":
    # 读取movieslen-1m数据集
    movies1m = pd.read_csv("../../dataset/ml-1m/ratings.dat", sep="::", names=["user", "item", "rate", "time"], nrows=None)
    movies1m.sort_values(by="time", ascending=True, inplace=True)
    data_lenth = len(movies1m)

    # 划分训练集和测试集
    train_data = movies1m[:int(data_lenth * 0.75)]
    test_data = movies1m[int(data_lenth * 0.75):]

    itemcf = ItemCF(train_data=train_data,
                    user_col_name="user",
                    item_col_name="item",
                    label_col_name="rate",
                    max_workers=1,
                    norm=True)

    k = 10
    n = 20
    print("测试集recall & precision: ")
    print(itemcf.evaluate(test_data, k, n))

    # 召回解释性
    print(set(train_data["user"].unique()).intersection(set(train_data["user"].unique())))
    movies_info = movies1m = pd.read_csv("../../dataset/ml-1m/movies.dat", sep="::", names=["item", "item_name", "tags"])

    print(f"每个trigger item取{k}个邻居, 共召回{n}个item.")
    users = [895, 6039, 5635]
    for user in users:
        pred_items = itemcf.recommend(user, k, n)
        # if not pred_items:
        #     continue
        pred_items = [item for item, score in pred_items]
        pred_items_names = movies_info[movies_info["item"].isin(list(pred_items))]
        his_items = itemcf.data[itemcf.data[itemcf.user_col_name] == user][itemcf.item_col_name]
        his_items_names = movies_info[movies_info["item"].isin(list(his_items))]
        print("="*200)
        print(f"user:{user}")
        print(f"历史行为items:\n{his_items_names['item_name']}")
        print("\n")
        print(f"召回items:\n{pred_items_names['item_name']}")

