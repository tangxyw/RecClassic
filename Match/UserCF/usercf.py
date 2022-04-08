"""
    UserCF的实现

    Reference:
    [1] 推荐系统实战 项亮
    [2] https://github.com/wangzhegeek/item-userCF
    [3] https://github.com/ZiyaoGeng/SimpleCF
    [4] https://github.com/xingzhexiaozhu/MovieRecommendation
    [5] https://github.com/xiaogp/recsys_spark

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


class UserCF:
    def __init__(self, train_data, user_col_name, item_col_name, label_col_name, max_workers=None):
        """

        Args:
            train_data (pd.DataFrame): 训练数据, 包含user, item, label三列, 且user, item已经由LabelEncoder编码完毕
            user_col_name (str): 用户列名
            item_col_name (str): 物料列名
            label_col_name (str): label列名
            max_workers (int): 线程数
        """

        self.data = train_data
        self.user_col_name = user_col_name
        self.item_col_name = item_col_name
        self.label_col_name = label_col_name
        self.max_workers = max_workers if max_workers else os.cpu_count() * 5
        # user数量
        self.n_user = train_data[user_col_name].nunique()
        # item数量
        self.n_item = train_data[item_col_name].nunique()

        # 多线程锁
        self.lock = threading.Lock()

        # user相似度矩阵
        print("正在生成user相似度矩阵...")
        self._get_user_similarity()
        print("生成完毕")

    def _get_item_user_index(self):
        """
            生成item-user的倒排索引, 以pd.DataFrame为容器
        Returns:
            item-user的倒排索引
        """

        item_user = self.data.groupby(self.item_col_name).apply(lambda x: list(x[self.user_col_name]))

        return item_user

    def _get_user_similarity(self):
        """
            生成user相似性字典,
        Returns:
            user相似性矩阵
        """

        # 每个user交互的item数量
        user_item_n = self.data.groupby(self.user_col_name).agg(
            items_n=pd.NamedAgg(column=self.user_col_name, aggfunc='count'))

        # 构建item-user的倒排索引
        item_user = self._get_item_user_index()

        # user相似度
        self.user_sim_dict = dict()

        def _calc_sim_partial(item_user):
            """
                得到user cos相似度的分子部分
            Args:
                item_user (pd.Series): item-user的倒排索引

            Returns:

            """

            for index, item_user_list in tqdm(item_user.iteritems(), total=item_user.count()):
                # item倒排索引列表长度
                item_user_list_length = len(item_user_list)
                # u,v为item_user_list的user索引下标
                for u in range(len(item_user_list) - 1):
                    user_u = item_user_list[u]
                    self.user_sim_dict.setdefault(user_u, {})
                    for v in range(u + 1, len(item_user_list)):
                        user_v = item_user_list[v]
                        self.user_sim_dict.setdefault(user_v, {})

                        self.user_sim_dict[user_u].setdefault(user_v, 0.0)
                        self.user_sim_dict[user_v].setdefault(user_u, 0.0)

                        # 热门惩罚
                        u_v_sim = 1.0 / math.log(1.0 + item_user_list_length)

                        # 多个线程写同一个对象, 需要全局锁
                        self.lock.acquire()
                        # 对称位置都要计算并更新
                        self.user_sim_dict[user_u][user_v] += u_v_sim
                        self.user_sim_dict[user_v][user_u] += u_v_sim
                        # 释放锁
                        self.lock.release()

        # 实际上由于锁的存在，多线程的效率不如单线程
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            # 每个线程处理的数据量
            sample_per_thead = self.n_item // self.max_workers + 1
            # 数据范围区间
            interval = []
            left_point = 0
            for i in range(self.max_workers):
                interval.append(item_user[left_point:left_point+sample_per_thead])
                left_point += sample_per_thead
                pool.map(_calc_sim_partial, interval)

        # 除以sqrt(N(u)*N(v)), 得到最终的cos相似度
        for user_u, related_users in self.user_sim_dict.items():
            for user_v, u_v_sim in related_users.items():
                self.user_sim_dict[user_u][user_v] /= math.sqrt(
                             user_item_n.at[user_u, "items_n"] * user_item_n.at[user_v, "items_n"])

    def recommend(self, user, k, n):
        """
            为一个user召回item
        Args:
            user (str): trigger
            k (int): 邻居user数量
            n: 召回item总数量

        Returns:
            召回item列表, [(item, score), ()...]
        """

        if user not in self.data[self.user_col_name].unique():
            print(f"{user}不在训练集中.")
            return

        # 召回item
        item_dict = dict()
        # user的历史交互item
        user_trigger_items = list(self.data[self.data[self.user_col_name] == user][self.item_col_name])

        # 遍历user最相似的k个邻居
        for neighbors_user, wij in sorted(self.user_sim_dict[user].items(), key=lambda x: x[1], reverse=True)[:k]:
            # 遍历每一个邻居的历史交互item作为possible_item
            for possible_item in self.data[self.data[self.user_col_name] == neighbors_user][self.item_col_name]:
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
            k (int): 邻居user数量
            n (int): 为一个user召回item的总数量

        Returns:
            整个测试数据集的recall, precision
        """

        recall, precision = 0, 0

        test_user_labels = test_data.groupby(self.user_col_name).apply(lambda x: list(x[self.item_col_name]))

        # 遍历测试集中每一个user
        print("开始评估测试集")
        for user, item_labels_list in tqdm(test_user_labels.iteritems(), total=test_user_labels.count()):
            if user in self.data[self.user_col_name].unique():
                # 召回items
                pred_items_list = [item for item, score in self.recommend(user, k, n)]
                user_recall, user_precision = MatchMetrics.get_recall_and_precision(labels=item_labels_list, preds=pred_items_list)
                recall += user_recall
                precision += user_precision

        # 测试集指标为每个user指标的算数平均
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

    usercf = UserCF(train_data=train_data,
                    user_col_name="user",
                    item_col_name="item",
                    label_col_name="rate",
                    max_workers=1)

    k = 10
    n = 20
    print("测试集recall & precision: ")
    print(usercf.evaluate(test_data, k, n))

    # 召回解释性
    # print(set(train_data["user"].unique()).intersection(set(train_data["user"].unique())))
    movies_info = movies1m = pd.read_csv("../../dataset/ml-1m/movies.dat", sep="::", names=["item", "item_name", "tags"])

    print(f"每个user取{k}个邻居, 召回{n}个item.")
    users = [895, 6039, 5635]
    for user in users:
        pred_items = usercf.recommend(user, k, n)
        if not pred_items:
            continue
        pred_items = [item for item, score in pred_items]
        pred_items_names = movies_info[movies_info["item"].isin(list(pred_items))]
        his_items = usercf.data[usercf.data[usercf.user_col_name] == user][usercf.item_col_name]
        his_items_names = movies_info[movies_info["item"].isin(list(his_items))]
        print("="*200)
        print(f"user:{user}")
        print(f"历史行为items:\n{his_items_names['item_name']}")
        print("\n")
        print(f"召回items:\n{pred_items_names['item_name']}")