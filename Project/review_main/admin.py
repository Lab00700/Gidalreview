from django.contrib import admin

from .models import eventModel

# Register your models here.

class reviewAdmin(admin.ModelAdmin):
    list_display = ("review_id", "review","date")

admin.site.register(eventModel, reviewAdmin)