from rest_framework import serializers
from .models import approval


class approvalSerializer(serializers.ModelSerializer):
    class Meta:
        model = approval
        fields = '__all__'