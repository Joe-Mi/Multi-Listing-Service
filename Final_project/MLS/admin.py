from django.contrib import admin
from .models import *

# Register your models here.
admin.site.register(User)
admin.site.register(Buyer)
admin.site.register(Seller)
admin.site.register(Listing)
admin.site.register(sellerProfile)
admin.site.register(buyerProfile)
admin.site.register(Chats)
admin.site.register(ChatLogs)
