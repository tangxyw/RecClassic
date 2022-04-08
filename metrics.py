


class MatchMetrics:

    @staticmethod
    def get_recall_and_precision(labels, preds, k=None):
        """
            计算召回model某个user的recall@k和precision@k
        Args:
            labels: user某个行为的ground truth items
            preds: 由召回model得到的items
            k (int): 取labels的前k个，如果为None，则取全部label

        Returns:
            user级别的recall@k和precision@k
        """

        if len(labels) == 0:
            return 0, 0

        if k:
            labels = labels[:k]

        # Pu ∩ Gu
        intersection = set(labels).intersection(set(preds))

        recall = len(intersection) / len(labels)
        precision = len(intersection) / len(preds)

        return recall, precision