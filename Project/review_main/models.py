from django.db import models

# Create your models here.

class eventModel(models.Model):
    review_id = models.CharField(max_length=256,primary_key=True)
    review = models.CharField(max_length=256)
    date = models.CharField(max_length=256, default='')
    label = models.CharField(max_length=256, default='')

    ################################################################# 추가
    star = models.CharField(max_length=10, default='')
    expectedPoint = models.FloatField(null=True, default=0)
    ################################################################# 추가

    class Meta:
        db_table = "review"

    def __str__(self):
        return self.review

class sorting(models.Model):
    class Meta:
        db_table = "sorting"
    review = models.CharField(max_length=256)
    date = models.DateField()
    expectedPoint = models.FloatField(null=True, default=0)
    similarComment1 = models.CharField(max_length=1000, default='')
    similarComment2 = models.CharField(max_length=1000, default='')
    label = models.CharField(max_length=256)
    star = models.IntegerField(null=True, default=0)
    comment = models.CharField(max_length=1000, default='')

class Rest_Info(models.Model):
    class Meta:
        db_table = "Rest_Info"
    star_point = models.FloatField(null=True, default=0)
    star = models.CharField(max_length=10, default='')
    expected_star_point = models.FloatField(null=True, default=0)
    expected_star = models.CharField(max_length=10, default='')
    rest_name = models.CharField(max_length=100, default='')
    review_notice = models.CharField(max_length=1000, default='')
    pic = models.CharField(max_length=1000, default='')


class crowling_all(models.Model):
    class Meta:
        db_table = "crowling_all"
    total = models.CharField(max_length=10, default='')
    taste = models.CharField(max_length=100, default='')
    quantity = models.CharField(max_length=1000, default='')
    delivery = models.CharField(max_length=1000, default='')
    com_name = models.CharField(max_length=100, default='')
    com_time = models.CharField(max_length=100, default='')
    order_menu = models.CharField(max_length=1000, default='')
    comment = models.CharField(max_length=1000, default='')
    com_picture_1 = models.CharField(max_length=1000, default='')
    com_picture_2 = models.CharField(max_length=1000, default='')
    com_picture_3 = models.CharField(max_length=1000, default='')


class SimilarComment(models.Model):
    class Meta:
        db_table = "SimilarComment"

    star = models.CharField(max_length=10, default='')
    comment = models.CharField(max_length=1000, default='')
    preprocessingComment = models.CharField(max_length=1000, default='')
    similarComment1 = models.CharField(max_length=1000, default='')
    similarity1 = models.FloatField(null=True, default=0)
    similarComment2 = models.CharField(max_length=1000, default='')
    similarity2 = models.FloatField(null=True, default=0)


# class restModel(models.Model):
#     class Meta:
#         db_table = "Rest_Info"
#
#     rest_name = models.CharField(max_length=256, default='',primary_key=True)
#     review_notice = models.CharField(max_length=256, default='')
#     r_notice_pic = models.CharField(max_length=256, default='')
#
#
# class ReviewModel(models.Model):
#     class Meta:
#         db_table = "review"
#
#     product_id = models.ForeignKey(ProductModel, on_delete=models.CASCADE, default='')
#     review = models.CharField(max_length=256)
#     score = models.FloatField(null=True, default=0)
#     keywords = models.CharField(max_length=256)
#     morph = models.CharField(max_length=256, default='')
#     xai_vale = models.CharField(max_length=256, default='')
#     date = models.CharField(max_length=256, default='')
#
# class ProductKeyword(models.Model):
#     class Meta:
#         db_table = "PDkeyword"
#
#     product_id = models.ForeignKey(ProductModel, on_delete=models.CASCADE, default='')
#     keyword = models.CharField(max_length=20, default='')
#     summarization = models.CharField(max_length=256, default='')
#     keyword_positive = models.FloatField(default=60)
