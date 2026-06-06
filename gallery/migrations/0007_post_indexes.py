from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('gallery', '0006_photo_is_video'),
    ]

    operations = [
        migrations.AlterField(
            model_name='post',
            name='rating',
            field=models.SmallIntegerField(default=0, db_index=True),
        ),
        migrations.AlterField(
            model_name='post',
            name='fav',
            field=models.BooleanField(default=False, db_index=True),
        ),
        migrations.AlterField(
            model_name='post',
            name='added_at',
            field=models.DateTimeField(auto_now_add=True, db_index=True),
        ),
    ]
