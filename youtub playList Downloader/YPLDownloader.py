from pytube import Playlist,YouTube
links=[]
# link = input("Enter YouTube Playlist url:")
OUTPUT="F:\\tot\\" # downloading path




for link in links:
    yt_playlist = Playlist(link)

    for video in yt_playlist.videos:
        out_path=OUTPUT+yt_playlist.title
        try:
            video.streams.get_by_resolution(resolution="720p").download(output_path=out_path)
            print(f" {video.title}: Done✔✔✔")
        except :
            try:
                video.streams.get_lowest_resolution().download(output_path=out_path)
                print(f" {video.title}: Done✔✔✔")
            except:
                print(f"video download field\n url: {video.title}")

            