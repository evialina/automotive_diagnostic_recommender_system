import { AppBar } from "react-admin"
import { Box, Typography } from '@mui/material';
const CustomAppBar = (props) => {
    return <AppBar {...props} sx={{ color: '#fff' }}>
        <Box component="span" flex={1} />
        <Typography variant='h4'>Diagnostic Action Recommender</Typography>
        <Box component="span" flex={1} />
        
    </AppBar>
}

export default CustomAppBar