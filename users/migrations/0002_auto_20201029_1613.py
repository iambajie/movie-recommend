# Generated by Django 3.1.1 on 2020-10-29 08:13

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='user',
            name='age',
            field=models.IntegerField(default=10),
        ),
        migrations.AddField(
            model_name='user',
            name='sex',
            field=models.CharField(default='男', max_length=10),
        ),
    ]